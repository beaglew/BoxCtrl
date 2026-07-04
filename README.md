# BoxCtrl

Official code for [BoxCtrl: 3D-Aware Visual Prompting for Geometric Image Editing](https://arxiv.org/abs/2606.23270) (SIGGRAPH 2026).

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Project Page](https://img.shields.io/badge/Project-Page-orange.svg)](https://beaglew.github.io/BoxCtrl-site/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/2606.23270)

Edit objects via translation, rotation, scaling, or combined transforms using RGB 3D bounding boxes as visual prompts. Built on [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev).

## ⚙️ Setup

```bash
git clone https://github.com/beaglew/BoxCtrl.git
cd BoxCtrl

export PYTHONNOUSERSITE=1
conda create -n boxctrl python=3.10
conda activate boxctrl

# 1) Install PyTorch for your CUDA — https://pytorch.org/get-started/previous-versions/
# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# 2) Install other required packages
pip install -r requirements.txt
```

## 📥 Download Models

**Base Model** — [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)

```bash
huggingface-cli login
huggingface-cli download black-forest-labs/FLUX.1-Kontext-dev --local-dir ./models/FLUX.1-Kontext-dev
```

**BoxCtrl LoRA** — [wakalaka/BoxCtrl](https://huggingface.co/wakalaka/BoxCtrl)

```bash
huggingface-cli download wakalaka/BoxCtrl --local-dir ./checkpoints/lora
```

## 🎯 Inference

Each sample in `assets/metadata.json` needs: `image_id`, `prompt`, `image`, `src_box`, `tgt_box`.

```bash
python infer.py \
  --base_path ./models/FLUX.1-Kontext-dev \
  --lora_path ./checkpoints/lora \
  --input_dir ./assets \
  --output_dir ./outputs \
  --json_path ./assets/metadata.json \
  --device 0 \
  --dtype bf16
```

## 📚 Citation

```bibtex
@article{wang2026boxctrl,
  title={BoxCtrl: 3D-Aware Visual Prompting for Geometric Image Editing},
  author={Wang, Feifei and Yang, Shiyuan and Li, Xiaoyu and Liao, Jing},
  journal={arXiv preprint arXiv:2606.23270},
  year={2026}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.

[FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) and [BoxCtrl LoRA](https://huggingface.co/wakalaka/BoxCtrl) weights are not included in this repo. Download them from Hugging Face and follow the [FLUX.1 [dev] Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev).

## 🙏 Acknowledgments

We thank the following open-source projects and research works that made BoxCtrl possible:

- **[FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)** by [Black Forest Labs](https://blackforestlabs.ai/) — our base image editing model
- **[Orient Anything V2](https://orient-anythingv2.github.io/)** ([SpatialVision/Orient-Anything-V2](https://github.com/SpatialVision/Orient-Anything-V2)) — unified 3D orientation and rotation understanding
- **[DiffusionNFT](https://github.com/NVlabs/DiffusionNFT)** — online diffusion reinforcement learning via forward-process optimization
- **[Flow-GRPO](https://github.com/yifan123/flow_grpo)** — online RL training framework for flow matching models
- **[Edit-R1](https://github.com/PKU-YuanGroup/Edit-R1)** — reinforcement learning framework for instruction-based image editing
- **[HuggingFace](https://huggingface.co/)** for the [diffusers](https://github.com/huggingface/diffusers) library
