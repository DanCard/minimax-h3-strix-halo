#!/usr/bin/env python3
"""Timing probe for MiniMax-H3 via a running ComfyUI server.

Builds the t2va graph directly in API format (the bundled template wraps
everything in a subgraph, which is awkward to drive programmatically) and times
the queue-to-completion latency.

ComfyUI keeps models resident between runs, so the FIRST run includes ~36GB of
weight loading and MIOpen kernel autotune. Subsequent runs against the same
server are close to pure generation time. Both are reported.

Note ComfyUI's `length` accepts the 17k+5 grid down to 5 frames -- it does NOT
enforce the 5-second floor the diffusers integration does. Values below 124 are
off-distribution (the model was trained at ~124-362) so quality will suffer, but
they run, which makes them useful for cheap timing.

Usage:
    python probe_comfy.py                          # 512x288, 124 frames, 4 steps
    python probe_comfy.py --length 22 --steps 4    # ~0.9s clip, fastest possible
    python probe_comfy.py --width 1344 --height 768 --steps 8 --tag full
"""

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

HOST = "http://127.0.0.1:8188"
HERE = Path(__file__).parent

UNET = "minimax_h3_fl2va_int8.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_int4.safetensors"
VAE_VIDEO = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUDIO = "minimax_h3_audio_vae_fp32.safetensors"
LORA_TURBO = "minimax_h3_turbo_4step_768p.safetensors"

DEFAULT_PROMPT = (
    "A red fox trotting through a snowy pine forest, snow crunching underfoot. "
    "Audio: soft wind through branches, paws compressing snow."
)


def snap_length(n):
    """ComfyUI's grid is 17k+5, minimum 5."""
    n = max(int(n), 5)
    if (n - 5) % 17 == 0:
        return n
    return 5 + 17 * (((n - 5) // 17) + 1)


def build_workflow(a):
    """API-format graph. Keys are node ids, values are {class_type, inputs}."""
    wf = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": CLIP, "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_VIDEO}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_AUDIO}},
        "6": {"class_type": "MiniMaxH3ImageToVideo",
              "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": a.prompt,
                         "width": a.width, "height": a.height, "length": a.length}},
        "7": {"class_type": "BasicGuider",
              "inputs": {"model": ["5", 0] if a.lora else ["1", 0], "conditioning": ["6", 0]}},
        "8": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": a.sampler}},
        "9": {"class_type": "BasicScheduler",
              "inputs": {"model": ["5", 0] if a.lora else ["1", 0],
                         "scheduler": a.scheduler, "steps": a.steps, "denoise": 1.0}},
        "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": a.seed}},
        "11": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["10", 0], "guider": ["7", 0], "sampler": ["8", 0],
                          "sigmas": ["9", 0], "latent_image": ["6", 1]}},
        "12": {"class_type": "VAEDecode", "inputs": {"samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["11", 0], "vae": ["4", 0]}},
        "14": {"class_type": "CreateVideo",
               "inputs": {"images": ["12", 0], "audio": ["13", 0], "fps": 24.0}},
        "15": {"class_type": "SaveVideo",
               "inputs": {"video": ["14", 0], "filename_prefix": f"h3-{a.tag}",
                          "format": "auto", "codec": "auto"}},
    }
    if a.lora:
        wf["5"] = {"class_type": "LoraLoaderModelOnly",
                   "inputs": {"model": ["1", 0], "lora_name": LORA_TURBO,
                              "strength_model": a.lora_strength}}
    return wf


def post(path, payload):
    req = urllib.request.Request(
        f"{HOST}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def get(path):
    with urllib.request.urlopen(f"{HOST}{path}", timeout=60) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=288)
    p.add_argument("--length", type=int, default=124, help="frames; snapped to 17k+5")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--tag", default="probe")
    p.add_argument("--sampler", default="res_multistep")
    p.add_argument("--scheduler", default="simple")
    p.add_argument("--lora", action="store_true", default=True, help="use 4-step turbo LoRA")
    p.add_argument("--no-lora", dest="lora", action="store_false")
    p.add_argument("--lora-strength", type=float, default=1.0)
    a = p.parse_args()

    for n, v in (("width", a.width), ("height", a.height)):
        if v % 32:
            raise SystemExit(f"--{n} must be a multiple of 32, got {v}")
    a.length = snap_length(a.length)

    try:
        get("/system_stats")
    except urllib.error.URLError as e:
        raise SystemExit(f"ComfyUI not reachable at {HOST}: {e}")

    print(f"[probe] {a.width}x{a.height}, {a.length} frames ({a.length/24:.2f}s), "
          f"{a.steps} steps, lora={'on' if a.lora else 'off'}")

    cid = str(uuid.uuid4())
    t0 = time.perf_counter()
    r = post("/prompt", {"prompt": build_workflow(a), "client_id": cid})
    pid = r["prompt_id"]
    print(f"[probe] queued {pid}")

    last = ""
    while True:
        time.sleep(2)
        hist = get(f"/history/{pid}")
        if pid in hist:
            break
        try:
            q = get("/queue")
            state = "running" if q.get("queue_running") else "pending"
            if state != last:
                print(f"[probe] {state}... ({time.perf_counter()-t0:.0f}s)")
                last = state
        except Exception:
            pass

    total = time.perf_counter() - t0
    entry = hist[pid]
    status = entry.get("status", {})
    if not status.get("completed", True):
        print(f"[probe] FAILED after {total:.0f}s")
        for m in status.get("messages", []):
            print("   ", m)
        raise SystemExit(1)

    outs = []
    for node_out in entry.get("outputs", {}).values():
        for k in ("images", "video", "gifs"):
            for f in node_out.get(k, []) or []:
                outs.append(f.get("filename"))

    print(f"\n[probe] TOTAL {total/60:.2f} min ({total:.0f}s)")
    print(f"[probe] {total/a.steps:.1f}s per step, "
          f"{total/(a.length/24):.0f}s per second of video")
    print(f"[probe] outputs: {outs}")

    row = {"utc": datetime.now(timezone.utc).isoformat(), "tag": a.tag,
           "width": a.width, "height": a.height, "length": a.length,
           "duration_s": round(a.length / 24, 2), "steps": a.steps,
           "lora": a.lora, "sampler": a.sampler, "scheduler": a.scheduler,
           "total_s": round(total, 1), "s_per_step": round(total / a.steps, 1),
           "outputs": outs}
    with open(HERE / "timings.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
