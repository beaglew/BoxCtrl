#!/usr/bin/env python3
"""Patch GroundingDINO for newer PyTorch / CUDA builds.

GroundingDINO issue #397 reports Torch 2.7 + CUDA 12.8 compilation
failures, and the same source incompatibility appears with Torch 2.6 +
CUDA 12.4. The fix is to use Tensor.scalar_type() and
Tensor.device().is_cuda() for newer PyTorch instead of the deprecated
Tensor.type() API.
"""

from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "GET_TENSOR_TYPE"


def patch_ms_deform_attn_cuda(groundingdino_root: Path) -> bool:
    target = (
        groundingdino_root
        / "groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu"
    )
    if not target.exists():
        raise FileNotFoundError(f"Cannot find {target}")

    text = target.read_text(encoding="utf-8")
    if PATCH_MARKER not in text:
        text = text.replace(
            "#include <cuda_runtime.h>\n",
            "#include <cuda_runtime.h>\n"
            "#include <torch/extension.h>\n"
            "#include <torch/version.h>\n"
            "\n"
            "#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 6\n"
            "  #define GET_TENSOR_TYPE(x) x.scalar_type()\n"
            "  #define IS_CUDA_TENSOR(x) x.device().is_cuda()\n"
            "#else\n"
            "  #define GET_TENSOR_TYPE(x) x.type()\n"
            "  #define IS_CUDA_TENSOR(x) x.type().is_cuda()\n"
            "#endif\n",
        )

    replacements = {
        "value.type().is_cuda()": "IS_CUDA_TENSOR(value)",
        "spatial_shapes.type().is_cuda()": "IS_CUDA_TENSOR(spatial_shapes)",
        "level_start_index.type().is_cuda()": "IS_CUDA_TENSOR(level_start_index)",
        "sampling_loc.type().is_cuda()": "IS_CUDA_TENSOR(sampling_loc)",
        "attn_weight.type().is_cuda()": "IS_CUDA_TENSOR(attn_weight)",
        "grad_output.type().is_cuda()": "IS_CUDA_TENSOR(grad_output)",
        'AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_forward_cuda"': (
            'AT_DISPATCH_FLOATING_TYPES(GET_TENSOR_TYPE(value), "ms_deform_attn_forward_cuda"'
        ),
        'AT_DISPATCH_FLOATING_TYPES(value.type(), "ms_deform_attn_backward_cuda"': (
            'AT_DISPATCH_FLOATING_TYPES(GET_TENSOR_TYPE(value), "ms_deform_attn_backward_cuda"'
        ),
    }
    before = text
    for old, new in replacements.items():
        text = text.replace(old, new)

    target.write_text(text, encoding="utf-8")
    return text != before


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch GroundingDINO for newer torch / CUDA builds.")
    parser.add_argument(
        "--groundingdino-root",
        required=True,
        type=Path,
        help="Path to the cloned GroundingDINO directory.",
    )
    args = parser.parse_args()

    changed = patch_ms_deform_attn_cuda(args.groundingdino_root.resolve())
    status = "patched" if changed else "already patched"
    print(f"GroundingDINO ms_deform_attn_cuda.cu: {status}")


if __name__ == "__main__":
    main()
