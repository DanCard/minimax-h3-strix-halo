#!/usr/bin/env python3
"""Repair the Abiray safetensors files, which are corrupt as published.

Every file in Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot has a marker string
appended past the end of its declared tensor data, e.g.

    \\nL2P_bypass_minimax_h3_audio_vae_fp32.safetensors_1785752000\\n

safetensors rejects this with "Error while deserializing header: incomplete
metadata, file not fully covered", so ComfyUI cannot load any of them.

Truncating to the declared end is a verified-correct repair: for the audio VAE,
where Comfy-Org/MiniMax-H3 publishes a clean copy of the same file, the
truncated Abiray file is byte-identical (sha256 match).

Writes repaired copies to models_fixed/ rather than editing the HF cache blobs,
so the hub's integrity bookkeeping stays intact.
"""

import json
import os
import struct
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

QUANT = "Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot"
OUT = Path(__file__).parent / "models_fixed"

# (repo, filename, output name)
TARGETS = [
    (QUANT, "MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors",
     "minimax_h3_fl2va_int8.safetensors"),
    (QUANT, "text_encoders/qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
     "qwen3vl_32b_minimax_h3_int4.safetensors"),
    (QUANT, "vae/minimax_h3_video_vae_fp16.safetensors",
     "minimax_h3_video_vae_fp16.safetensors"),
]

CHUNK = 8 << 20


def declared_end(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    end = max(v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__")
    return 8 + n + end


def repair(src, dst):
    want = declared_end(src)
    have = os.path.getsize(src)
    if have == want:
        print(f"    already clean ({have} bytes)")
    else:
        print(f"    trimming {have - want} trailing bytes")
    written = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while written < want:
            b = fi.read(min(CHUNK, want - written))
            if not b:
                break
            fo.write(b)
            written += len(b)
    assert os.path.getsize(dst) == want, "short write"
    return want


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for repo, fn, out in TARGETS:
        src = hf_hub_download(repo_id=repo, filename=fn, local_files_only=True)
        dst = OUT / out
        print(f"--> {out}")
        if dst.exists() and os.path.getsize(dst) == declared_end(src):
            print("    up to date, skipping")
            continue
        n = repair(src, dst)
        print(f"    wrote {n/1e9:.3f} GB")

    print("\nverifying all repaired files load:")
    from safetensors.torch import safe_open
    for _, _, out in TARGETS:
        p = OUT / out
        with safe_open(p, framework="pt") as f:
            k = len(f.keys())
        print(f"    OK {k:5d} tensors  {out}")
    sys.stdout.flush()
