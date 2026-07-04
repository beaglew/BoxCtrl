#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import random
import json
from PIL import Image
from pathlib import Path
import argparse
import numpy as np
from pipeline import FluxKontextPipeline

def to_device_string(dev_arg) -> str:
    s = str(dev_arg)
    if s.isdigit():
        return f"cuda:{s}"
    return s 

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    ap = argparse.ArgumentParser(description="BoxCtrl inference")
    ap.add_argument(
        "--base_path",
        default="./models/FLUX.1-Kontext-dev",
        type=str,
        help="Path to FLUX.1-Kontext-dev base model (default: ./models/FLUX.1-Kontext-dev)",
    )
    ap.add_argument(
        "--lora_path",
        default="./checkpoints/lora",
        type=str,
        help="Path to BoxCtrl LoRA weights (default: ./checkpoints/lora)",
    )
    ap.add_argument(
        "--input_dir",
        default="./assets",
        type=str,
        help="Root directory for input images referenced in metadata (default: ./assets)",
    )
    ap.add_argument(
        "--output_dir",
        default="./outputs",
        type=str,
        help="Directory to save output images (default: ./outputs)",
    )
    ap.add_argument(
        "--json_path",
        default="./assets/metadata.json",
        type=str,
        help="Path to metadata JSON (default: ./assets/metadata.json)",
    )
    ap.add_argument(
        "--device",
        default="0",
        type=str,
        help="CUDA device id or device string, e.g. 0 or cuda:0 (default: 0)",
    )
    ap.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Model weight dtype (default: bf16)",
    )
    ap.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed (default: 42)",
    )
    ap.add_argument(
        "--guidance_scale",
        default=2.5,
        type=float,
        help="Classifier-free guidance scale (default: 2.5)",
    )
    ap.add_argument(
        "--height",
        default=1024,
        type=int,
        help="Output image height (default: 1024)",
    )
    ap.add_argument(
        "--width",
        default=1024,
        type=int,
        help="Output image width (default: 1024)",
    )
    ap.add_argument(
        "--num_inference_steps",
        default=28,
        type=int,
        help="Number of diffusion steps (default: 28)",
    )
    args = ap.parse_args()
    seed_everything(args.seed)

    device_str = to_device_string(args.device)
    if device_str.startswith("cuda"):
        torch.cuda.set_device(device_str)
    if args.dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)


    with open(args.json_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    pipeline = FluxKontextPipeline.from_pretrained(
        args.base_path,
        torch_dtype=weight_dtype,
    )
    pipeline.load_lora_weights(args.lora_path)
    pipeline.to(device_str)
    pipeline.transformer.eval()


    with torch.inference_mode():
        total = len(test_data)
        for idx, data in enumerate(test_data, start=1):
            image_id = data["image_id"]
            image_path = input_dir / data["image"]
            src_box_path = input_dir / data["src_box"]
            tgt_box_path = input_dir / data["tgt_box"]
            sample_dir = output_root / image_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            src_out = sample_dir / "src.png"
            pred_out = sample_dir / "pred.png"
            prompt = data["prompt"]

            print(f"[{idx}/{total}] {image_id} -> {pred_out}")

            generator = torch.Generator(device=device_str).manual_seed(args.seed)

            image_1 = Image.open(image_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
            src_box = Image.open(src_box_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
            gt_box  = Image.open(tgt_box_path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
            image_1.save(src_out)
 
            out = pipeline(
                image_1=image_1,
                image_2=src_box,
                image_3=gt_box,
                prompt=prompt,
                guidance_scale=args.guidance_scale,
                generator=generator,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
            )
            out.images[0].save(pred_out)

    print("[DONE] finished.")

if __name__ == "__main__":
    main()
