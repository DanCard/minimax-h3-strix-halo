#!/usr/bin/env python3
"""Fetch only the t2va components of MiniMax-H3 into the default HF cache.

The full repo is ~498GB of largely duplicate variants. The t2va workflow needs
the diffusers-format transformer, the Qwen3-VL conditioner, the two VAEs and the
small config/tokenizer folders — about 144GB. transformer_ref/, FL2VA/ and
Ref2VA/ are deliberately excluded.

Resumable: re-run after an interruption and it picks up where it stopped.
"""

import sys

from huggingface_hub import snapshot_download

REPO = "MiniMaxAI/MiniMax-H3"

ALLOW = [
    "model_index.json",
    "modular_model_index.json",
    "transformer/*",
    "text_encoder/*",
    "vae/*",
    "audio_vae/*",
    "tokenizer/*",
    "processor/*",
    "scheduler/*",
    "audio_scheduler/*",
]

if __name__ == "__main__":
    path = snapshot_download(
        repo_id=REPO,
        allow_patterns=ALLOW,
        max_workers=8,
        resume_download=True,
    )
    print(f"\nsnapshot: {path}")
    sys.stdout.flush()
