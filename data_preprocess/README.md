# 🧰 Data Preparation

Generate BoxCtrl inference inputs from a single image using [DetAny3D](https://github.com/OpenDriveLab/DetAny3D):

| Output | BoxCtrl field |
|--------|---------------|
| `src.png` | `image` in `metadata.json` |
| `src_rgb_box.png` | `src_box` |
| `gt_rgb_box.png` | `tgt_box` |
| `prompt.txt` | `prompt` |

DetAny3D source and checkpoints are not bundled here. Clone into `third_party/DetAny3D/` and download weights (below).

## ⚙️ Environment

```bash
cd data_preprocess   # inside BoxCtrl repo
conda create -n detany3d python=3.8
conda activate detany3d
python -m pip install -U pip wheel setuptools
```

## 🚀 Install

```bash
# 1. DetAny3D
git clone https://github.com/OpenDriveLab/DetAny3D.git third_party/DetAny3D

# 2. PyTorch (CUDA 12.1 example)
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121

# 3. SAM
pip install git+https://github.com/facebookresearch/segment-anything.git

# 4. GroundingDINO
git clone https://github.com/IDEA-Research/GroundingDINO.git third_party/DetAny3D/GroundingDINO
pip install -e third_party/DetAny3D/GroundingDINO --no-build-isolation --no-deps

# 5. Other deps
pip install -r requirements.txt
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html
```

Use the matching PyTorch/CUDA wheels for your setup — see [mmcv install guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

If GroundingDINO fails to build (e.g. newer PyTorch / CUDA), try the compatibility patch first:

```bash
python patch_groundingdino_torch26.py --groundingdino-root third_party/DetAny3D/GroundingDINO
pip install -e third_party/DetAny3D/GroundingDINO --no-build-isolation --no-deps
```

## 📦 Checkpoints

Place under `third_party/DetAny3D/`:

```
third_party/DetAny3D/
├── checkpoints/
│   ├── sam_ckpts/sam_vit_h_4b8939.pth
│   ├── unidepth_ckpts/model.pth
│   ├── dino_ckpts/dinov2_vitl14_pretrain.pth
│   └── detany3d_ckpts/other_exp_ckpt.pth
└── GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
```

- **SAM** `sam_vit_h_4b8939.pth`: [SAM GitHub Releases](https://github.com/facebookresearch/segment-anything)
- **UniDepth / DINO / DetAny3D** (`model.pth`, `dinov2_vitl14_pretrain.pth`, `other_exp_ckpt.pth`): [Google Drive](https://drive.google.com/drive/folders/17AOq5i1pCTxYzyqb1zbVevPy5jAXdNho?usp=drive_link)
- **GroundingDINO** `groundingdino_swinb_cogcoor.pth`: [official repo](https://github.com/IDEA-Research/GroundingDINO) → `GroundingDINO/weights/`

## 🔍 Run

With `--mask`, the object bbox comes from the mask; without it, GroundingDINO uses `--label`.

```bash
python detect_and_render_boxes.py \
  --detany3d-root third_party/DetAny3D \
  --image images/car.png \
  --out-dir outputs/car \
  --mask images/car_mask.png \
  --label "toy car" \
  --dyaw 60.0 \
  --device cuda:0
```

Without mask:

```bash
python detect_and_render_boxes.py \
  --detany3d-root third_party/DetAny3D \
  --image images/car.png \
  --out-dir outputs/car \
  --label "toy car" \
  --dyaw 60.0 \
  --device cuda:0
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--detany3d-root` | — | DetAny3D repo root |
| `--image` | — | Input image |
| `--out-dir` | — | Output directory |
| `--mask` | — | Mask image (bbox from mask) |
| `--label` | — | Object name for text detection and prompt |
| `--dx` / `--dy` | `0` | Translation (fraction of image width/height) |
| `--dyaw` | `0` | Yaw in degrees; `>0` clockwise → prompt **left**, `<0` → **right** |
| `--dscale` | `1.0` | Scale factor |
| `--rgb-size` | `1024` | Output PNG side length |
| `--device` | `cuda:0` | Torch device |

## 📚 Acknowledgments

[DetAny3D](https://github.com/OpenDriveLab/DetAny3D) — *Detect Anything 3D in the Wild* ([arXiv:2504.07958](https://arxiv.org/abs/2504.07958)); also [SAM](https://github.com/facebookresearch/segment-anything) and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO).

```bibtex
@inproceedings{zhang2025detect,
  title={Detect anything 3d in the wild},
  author={Zhang, Hanxue and Jiang, Haoran and Yao, Qingsong and Sun, Yanan and Zhang, Renrui and Zhao, Hao and Li, Hongyang and Zhu, Hongzi and Yang, Zetong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5048--5059},
  year={2025}
}
```
