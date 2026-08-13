#!/usr/bin/env python3
"""Fetch pre-quantized MiniMax-H3 weights (ComfyUI layout) + the 4-step turbo LoRA.

Replaces the 144GB BF16 diffusers download. These are already quantized, so there
is no load-time spike from materializing BF16 and quantizing in place.

FL2VA covers text-to-video and image-to-video. Ref2VA (omni-references) is not
fetched — add it later if needed.

The transformer is int8 (the highest-quality pruned variant, ~21GB). The repo's
guide recommends an int8 text encoder too, but only int4 and nvfp4 text encoders
actually exist in the repo, so int4 it is.

Note these are *pruned* community conversions, not official weights. Quality is
unverified against the full model.

Total ~44GB.
"""

import sys

from huggingface_hub import hf_hub_download, snapshot_download

QUANT_REPO = "Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot"
QUANT_FILES = [
    "MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors",   # ~21.0 GB
    "text_encoders/qwen3vl_32b_minimax_h3_int4_convrot.safetensors",  # ~15.0 GB
    "vae/minimax_h3_video_vae_fp16.safetensors",          # ~5.2 GB
    "vae/minimax_h3_audio_vae_fp32.safetensors",          # ~0.6 GB
]

TURBO_REPO = "lightx2v/Minimax-h3-Turbo"
TURBO_FILE = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"  # ~2.0 GB


if __name__ == "__main__":
    for name in QUANT_FILES:
        print(f"--> {name}", flush=True)
        p = hf_hub_download(repo_id=QUANT_REPO, filename=name)
        print(f"    {p}", flush=True)

    print(f"--> {TURBO_FILE}", flush=True)
    p = hf_hub_download(repo_id=TURBO_REPO, filename=TURBO_FILE)
    print(f"    {p}", flush=True)

    print("\nall quantized weights fetched", flush=True)
    sys.stdout.flush()
