#!/usr/bin/env python3
"""Generate BoxCtrl inputs: src.png, src_rgb_box.png, gt_rgb_box.png, prompt.txt.

Example (with mask):
  python detect_and_render_boxes.py \\
    --detany3d-root third_party/DetAny3D \\
    --image images/car.png \\
    --out-dir outputs/car \\
    --mask images/car_mask.png \\
    --label "toy car" \\
    --dyaw 60.0 \\
    --device cuda:0

Without mask, omit --mask and use --label for GroundingDINO.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

cv2 = None
np = None
torch = None
F = None
yaml = None
Box = None
Image = None
box_convert = None


def import_runtime_deps() -> None:
    global cv2, np, torch, F, yaml, Box, Image, box_convert

    import cv2 as _cv2
    import numpy as _np
    import torch as _torch
    import torch.nn.functional as _F
    import yaml as _yaml
    from box import Box as _Box
    from PIL import Image as _Image
    from torchvision.ops import box_convert as _box_convert

    cv2 = _cv2
    np = _np
    torch = _torch
    F = _F
    yaml = _yaml
    Box = _Box
    Image = _Image
    box_convert = _box_convert


def add_detany3d_to_path(detany3d_root: Path) -> None:
    root = str(detany3d_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    grounding = detany3d_root / "GroundingDINO"
    if grounding.exists() and str(grounding.resolve()) not in sys.path:
        sys.path.insert(0, str(grounding.resolve()))


def disable_distributed() -> None:
    torch.distributed.is_available = lambda: False
    torch.distributed.is_initialized = lambda: False
    torch.distributed.get_world_size = lambda group=None: 1
    torch.distributed.get_rank = lambda group=None: 0


def preprocess_sam(x: torch.Tensor, cfg: Box) -> torch.Tensor:
    mean = torch.Tensor(cfg.dataset.pixel_mean).view(-1, 1, 1)
    std = torch.Tensor(cfg.dataset.pixel_std).view(-1, 1, 1)
    x = (x - mean) / std
    h, w = x.shape[-2:]
    return F.pad(x, (0, cfg.model.pad - w, 0, cfg.model.pad - h))


def preprocess_dino(x: torch.Tensor) -> torch.Tensor:
    x = x / 255
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (x - mean) / std


def crop_hw(img: torch.Tensor) -> torch.Tensor:
    if img.dim() == 4:
        img = img.squeeze(0)
    h, w = img.shape[1:3]
    new_h, new_w = (h // 14) * 14, (w // 14) * 14
    ch, cw = h // 2, w // 2
    return img[:, ch - new_h // 2 : ch - new_h // 2 + new_h, cw - new_w // 2 : cw - new_w // 2 + new_w].unsqueeze(0)


def rotation_matrix_y(theta_rad: float) -> np.ndarray:
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


def vertices_to_original_coords(verts: np.ndarray, scale_w: float, scale_h: float) -> np.ndarray:
    out = verts.copy()
    out[:, 0] /= scale_w
    out[:, 1] /= scale_h
    return out


def draw_faces_on_image(img_rgb: np.ndarray, verts_2d: np.ndarray) -> np.ndarray:
    for idxs, color in (
        ([0, 1, 5, 4], (0, 255, 0)),   # side — green
        ([3, 0, 4, 7], (0, 0, 255)),   # top — blue
        ([0, 1, 2, 3], (255, 0, 0)),   # front — red (on top)
    ):
        pts = np.array(verts_2d[idxs], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillConvexPoly(img_rgb, pts, color)
    return img_rgb


def draw_bbox_2d_wireframe(image: np.ndarray, points_2d: np.ndarray, thickness: int = 2) -> None:
    pts = [tuple(map(int, p)) for p in points_2d.astype(int)]
    for i in range(4):
        cv2.line(image, pts[i], pts[(i + 1) % 4], (0, 0, 0), thickness)
    for i in range(4, 8):
        cv2.line(image, pts[i], pts[(i + 1) % 4 + 4], (0, 0, 0), thickness)
    for i in range(4):
        cv2.line(image, pts[i], pts[i + 4], (0, 0, 0), thickness)


def load_binary_mask(mask_path: Path, img_w: int, img_h: int) -> np.ndarray:
    img = Image.open(mask_path)
    if img.mode not in ("L", "RGB", "RGBA"):
        img = img.convert("RGBA")
    arr = np.array(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        mask = (arr[:, :, 3] > 127).astype(np.uint8)
        if mask.sum() == 0:
            mask = (arr[:, :, :3].max(axis=2) > 127).astype(np.uint8)
    elif arr.ndim == 3:
        mask = (arr.max(axis=2) > 127).astype(np.uint8)
    else:
        mask = (arr > 127).astype(np.uint8)
    if mask.shape != (img_h, img_w):
        mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    return mask


def box_xyxy_from_mask(mask_path: Path, img_w: int, img_h: int) -> list[int]:
    mask = load_binary_mask(mask_path, img_w, img_h)
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        raise RuntimeError(f"Empty mask: {mask_path}")
    xmin, ymin = int(cols.min()), int(rows.min())
    xmax, ymax = int(cols.max()), int(rows.max())
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_w - 1, xmax)
    ymax = min(img_h - 1, ymax)
    if xmax - xmin < 32:
        cx = (xmin + xmax) // 2
        xmin, xmax = max(0, cx - 16), min(img_w - 1, cx + 16)
    if ymax - ymin < 32:
        cy = (ymin + ymax) // 2
        ymin, ymax = max(0, cy - 16), min(img_h - 1, cy + 16)
    return [xmin, ymin, xmax, ymax]


def label_candidates(label: str) -> list[str]:
    label = label.strip()
    if not label:
        return []
    out = [label]
    low = label.lower()
    if not (low.startswith("a ") or low.startswith("an ")):
        article = "an" if label[0].lower() in "aeiou" else "a"
        out.append(f"{article} {label}")
    return out


def build_prompt(label: str, dx: float, dy: float, dyaw: float, dscale: float) -> str:
    """Build edit prompt. Rotation: dyaw>0 → clockwise → 'left'; dyaw<0 → 'right'."""
    label = label.strip() or "object"
    parts: list[str] = []

    if abs(dyaw) < 1e-6:
        parts.append(f"keep the {label} orientation unchanged")
    elif dyaw > 0:
        parts.append(f"turn the {label} left {abs(dyaw):.1f} degrees")   # clockwise
    else:
        parts.append(f"turn the {label} right {abs(dyaw):.1f} degrees")

    if abs(dscale - 1.0) < 1e-6:
        parts.append(f"keep the {label} size unchanged")
    elif dscale > 1.0:
        parts.append(f"enlarge the {label} by a factor of {dscale:.2f}")
    else:
        parts.append(f"shrink the {label} by a factor of {dscale:.2f}")

    move_dirs: list[str] = []
    if abs(dx) >= 1e-6:
        move_dirs.append("right" if dx > 0 else "left")
    if abs(dy) >= 0.1:
        move_dirs.append("down" if dy > 0 else "up")
    if move_dirs:
        parts.append(f"move the {label} {' and '.join(move_dirs)}")
    else:
        parts.append(f"keep the {label} position unchanged")

    return ", ".join(parts)


def save_src_image(image_path: Path, out_path: Path, out_size: int) -> None:
    img = Image.open(image_path).convert("RGB")
    if out_size != img.width or out_size != img.height:
        img = img.resize((out_size, out_size), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def render_white_3dbox(
    h: int, w: int, verts: np.ndarray, out_path: Path, out_size: int = 1024,
) -> None:
    canvas = draw_faces_on_image(np.full((h, w, 3), 255, dtype=np.uint8), verts)
    draw_bbox_2d_wireframe(canvas, verts)
    if out_size != w or out_size != h:
        canvas = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


def compute_target_vertices_2d(
    bbox_3d, rot_mat, K, img_h, img_w, scale_w, scale_h,
    dyaw_deg, dx, dy, scale, vertex_fn, project_fn,
) -> np.ndarray:
    x, y, z, w3d, h3d, l3d, yaw = bbox_3d
    rot = rot_mat @ rotation_matrix_y(math.radians(-dyaw_deg))
    v3d, _ = vertex_fn(x, y, z, w3d, h3d, l3d, yaw, rot)
    v2d = vertices_to_original_coords(project_fn(v3d, K), scale_w, scale_h)
    center = v2d.mean(axis=0)
    v2d = (v2d - center) * scale + center
    v2d[:, 0] += dx * img_w
    v2d[:, 1] += dy * img_h
    return v2d


def load_models(detany3d_root: Path, config_path: Path, device: str):
    add_detany3d_to_path(detany3d_root)
    os.chdir(detany3d_root)
    disable_distributed()

    from groundingdino.util.inference import load_model, predict as gd_predict
    import groundingdino.datasets.transforms as T
    from train_utils import (
        ResizeLongestSide, compute_3d_bbox_vertices, decode_bboxes,
        project_to_image, rotation_6d_to_matrix,
    )
    from wrap_model import WrapModel

    with open(config_path, encoding="utf-8") as f:
        cfg = Box(yaml.load(f.read(), Loader=yaml.FullLoader))

    model = WrapModel(cfg)
    ckpt = torch.load(cfg.resume, map_location=device)
    state = model.state_dict()
    for k, v in state.items():
        if k in ckpt["state_dict"] and ckpt["state_dict"][k].size() == v.size():
            state[k] = ckpt["state_dict"][k].detach()
    model.load_state_dict(state)
    model.to(device)
    model.setup()
    model.eval()

    gd = load_model(
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        "GroundingDINO/weights/groundingdino_swinb_cogcoor.pth",
    ).eval()

    return cfg, model, gd, {
        "sam_trans_cls": ResizeLongestSide,
        "vertex_fn": compute_3d_bbox_vertices,
        "decode_bboxes": decode_bboxes,
        "project_fn": project_to_image,
        "rot_fn": rotation_6d_to_matrix,
        "T": T,
        "gd_predict": gd_predict,
    }


def detect(
    image_path: Path,
    device: str,
    cfg, model, gd_model, h, sam_trans,
    *,
    mask_path: Path | None,
    label: str,
) -> tuple[list[dict], dict, str]:
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)
    img_h, img_w = img_np.shape[:2]

    if mask_path is not None:
        boxes = [box_xyxy_from_mask(mask_path, img_w, img_h)]
        names = [label]
        det_mode = f"mask:{mask_path.name}"
    else:
        candidates = label_candidates(label)
        if not candidates:
            raise RuntimeError("Need --label for text detection")
        last_err = None
        boxes = names = None
        det_mode = ""
        for cap in candidates:
            try:
                tfm = h["T"].Compose([
                    h["T"].RandomResize([800], max_size=1333),
                    h["T"].ToTensor(),
                    h["T"].Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                img_gd = tfm(Image.fromarray(img_np), None)[0]
                b, _, phrases = h["gd_predict"](
                    model=gd_model, image=img_gd, caption=cap,
                    box_threshold=0.37, text_threshold=0.25, remove_combined=False,
                )
                b = b * torch.Tensor([img_w, img_h, img_w, img_h])
                xyxy = box_convert(boxes=b, in_fmt="cxcywh", out_fmt="xyxy")
                boxes = [x.to(torch.int).cpu().numpy().tolist() for x in xyxy]
                names = list(phrases)
                if not boxes:
                    raise RuntimeError("No objects found")
                det_mode = f"text:{cap}"
                break
            except RuntimeError as e:
                if "No objects found" not in str(e):
                    raise
                last_err = e
        if boxes is None:
            raise RuntimeError(last_err or "No objects found")

    t_img = crop_hw(sam_trans.apply_image_torch(
        torch.from_numpy(img_np).permute(2, 0, 1).float().unsqueeze(0)
    ))
    bh, bw = int(t_img.shape[2]), int(t_img.shape[3])
    if cfg.model.vit_pad_mask:
        vps = (bh // cfg.model.image_encoder.patch_size, bw // cfg.model.image_encoder.patch_size)
    else:
        vps = (cfg.model.pad // cfg.model.image_encoder.patch_size,) * 2

    box_t = sam_trans.apply_boxes_torch(torch.tensor(boxes), (img_h, img_w)).to(torch.int).to(device)
    with torch.no_grad():
        ret = model({
            "images": preprocess_sam(t_img, cfg).to(device),
            "vit_pad_size": torch.tensor(vps).to(device).unsqueeze(0),
            "images_shape": torch.Tensor((bh, bw)).to(device).unsqueeze(0),
            "image_for_dino": preprocess_dino(t_img).to(device),
            "boxes_coords": box_t,
        })
        K = ret["pred_K"]
        _, b3d = h["decode_bboxes"](ret, cfg, K)
        R = h["rot_fn"](ret["pred_pose_6d"])

    K_np = K.detach().cpu().numpy()
    sh, sw = bh / img_h, bw / img_w
    objects = []
    for i in range(len(b3d)):
        x, y, z, w3d, h3d, l3d, yaw = b3d[i].detach().cpu().numpy()
        Ri = R[i].detach().cpu().numpy()
        v3d, _ = h["vertex_fn"](x, y, z, w3d, h3d, l3d, yaw, Ri)
        v2d = h["project_fn"](v3d, K_np.squeeze(0))
        objects.append({
            "label": str(names[i]),
            "bbox_3d": [float(x), float(y), float(z), float(w3d), float(h3d), float(l3d), float(yaw)],
            "vertices_2d_orig": vertices_to_original_coords(v2d, sw, sh).tolist(),
            "rot_mat": Ri.tolist(),
            "K": K_np.squeeze(0).tolist(),
        })

    scene = {"img_h": img_h, "img_w": img_w, "scale_h": sh, "scale_w": sw,
             "vertex_fn": h["vertex_fn"], "project_fn": h["project_fn"]}
    return objects, scene, det_mode


def main() -> None:
    p = argparse.ArgumentParser(description="Detect one image -> src/gt RGB boxes")
    p.add_argument("--detany3d-root", required=True, type=Path)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--mask", type=Path, default=None, help="Mask image (L/RGBA)")
    p.add_argument("--label", default="", help="Text prompt when --mask is not given")
    p.add_argument("--dx", type=float, default=0.0)
    p.add_argument("--dy", type=float, default=0.0)
    p.add_argument("--dyaw", type=float, default=0.0)
    p.add_argument("--dscale", type=float, default=1.0)
    p.add_argument("--rgb-size", type=int, default=1024, help="Output image size (default: 1024)")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    import_runtime_deps()

    root = args.detany3d_root.resolve()
    cfg_path = args.config.resolve() if args.config else root / "detect_anything/configs/demo.yaml"
    image = args.image.expanduser().resolve()
    out = args.out_dir.expanduser().resolve()
    mask = args.mask.expanduser().resolve() if args.mask else None
    out.mkdir(parents=True, exist_ok=True)

    if not image.exists():
        p.error(f"Image not found: {image}")
    if mask is None and not args.label.strip():
        p.error("Need --mask or --label")

    cfg, model, gd, h = load_models(root, cfg_path, args.device)
    sam_trans = h["sam_trans_cls"](cfg.model.pad)
    objects, scene, det_mode = detect(
        image, args.device, cfg, model, gd, h, sam_trans,
        mask_path=mask,
        label=args.label.strip() or "object",
    )

    if not objects:
        raise RuntimeError("No objects detected")

    obj = objects[0]
    src = np.array(obj["vertices_2d_orig"], dtype=np.float32)
    tgt = compute_target_vertices_2d(
        obj["bbox_3d"], np.array(obj["rot_mat"]), np.array(obj["K"]),
        scene["img_h"], scene["img_w"], scene["scale_w"], scene["scale_h"],
        args.dyaw, args.dx, args.dy, args.dscale,
        scene["vertex_fn"], scene["project_fn"],
    )

    src_image_path = out / "src.png"
    src_path = out / "src_rgb_box.png"
    gt_path = out / "gt_rgb_box.png"
    save_src_image(image, src_image_path, args.rgb_size)
    render_white_3dbox(scene["img_h"], scene["img_w"], src, src_path, args.rgb_size)
    render_white_3dbox(scene["img_h"], scene["img_w"], tgt, gt_path, args.rgb_size)

    label = args.label.strip() or obj["label"]
    prompt = build_prompt(label, args.dx, args.dy, args.dyaw, args.dscale)
    prompt_path = out / "prompt.txt"
    prompt_path.write_text(prompt + "\n", encoding="utf-8")

    print(json.dumps({
        "label": label,
        "prompt": prompt,
        "detection": det_mode,
        "src_png": str(src_image_path),
        "src_rgb_box": str(src_path),
        "gt_rgb_box": str(gt_path),
        "prompt_file": str(prompt_path),
    }, indent=2))


if __name__ == "__main__":
    main()
