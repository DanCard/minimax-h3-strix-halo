#!/usr/bin/env python3
"""Timing probe for MiniMax-H3 on Strix Halo (gfx1151).

UNTESTED. Nothing in this file has been run against real weights yet — it is the
documented diffusers recipe adapted for this machine. Expect to debug it.

Purpose is to answer one question: how many minutes does one clip take on this
box? It deliberately runs at a small canvas, where output will look bad. That is
fine — this measures speed, not quality.

Constraints worth knowing before you touch the arguments (see README):
  * num_frames snaps up to the next 17n+5 and the duration must land in 5-15s,
    so 124 frames (~5.17s) is the hard floor. Shorter clips are impossible.
  * The checkpoint is guidance-distilled, so there is no guidance_scale to tune
    and every step is a single forward pass.
  * Canvas is the big lever: 960x544 is ~2.3x faster per step than the trained
    1344x768, and dimensions only have to be multiples of 32.

Usage:
    python run_probe.py --width 512 --height 288 --steps 8
    python run_probe.py --width 960 --height 544 --steps 16 --tag baseline
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Must be set before torch initialises ROCr. Disabling SVM management avoids
# thrashing on unified memory and was worth ~3x on Hunyuan Video on gfx1151.
os.environ.setdefault("HSA_USE_SVM", "0")

import torch  # noqa: E402

REPO = "MiniMaxAI/MiniMax-H3"
FPS = 24
MIN_FRAMES = 124  # 17*7+5, the shortest the video VAE can decode above the 5s floor
HERE = Path(__file__).parent
DEFAULT_PROMPT = "A red fox trotting through a snowy pine forest, snow crunching underfoot"


def snap_frames(n):
    """Round up to the next 17n+5 the video VAE can decode."""
    n = max(n, MIN_FRAMES)
    if (n - 5) % 17 == 0:
        return n
    return 5 + 17 * (((n - 5) // 17) + 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=int, default=512, help="multiple of 32 (default: 512)")
    p.add_argument("--height", type=int, default=288, help="multiple of 32 (default: 288)")
    p.add_argument("--frames", type=int, default=MIN_FRAMES, help=f"snapped to 17n+5, min {MIN_FRAMES}")
    p.add_argument("--steps", type=int, default=8, help="num_inference_steps (default: 8)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="probe", help="label for the output file and timings row")
    p.add_argument("--precision", choices=["int8", "bf16"], default="int8",
                   help="bf16 needs ~124GB and will not fit alongside anything else")
    p.add_argument("--no-stream", action="store_true",
                   help="disable streamed offload if it misbehaves on ROCm")
    return p.parse_args()


def build_pipeline(args, device):
    from diffusers import ComponentsManager, ModularPipeline

    if args.precision == "bf16":
        # 61.7GB transformer + 62.1GB conditioner against 124GB of RAM. Only
        # viable with the manager evicting aggressively, and probably not then.
        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device=device, memory_reserve_margin="12GB")
        pipe = ModularPipeline.from_pretrained(REPO, components_manager=manager)
        pipe.load_components(workflow="t2va", dtype=torch.bfloat16)
        return pipe

    from diffusers import MiniMaxH3Transformer3DModel, TorchAoConfig
    from diffusers.hooks import apply_group_offloading
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    pipe = ModularPipeline.from_pretrained(REPO)
    pipe.update_components(
        transformer=MiniMaxH3Transformer3DModel.from_pretrained(
            REPO, subfolder="transformer", dtype=torch.bfloat16,
            quantization_config=TorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=[
                    "proj_in", "audio_proj_in", "context_embedder", "time_embedder",
                    "time_proj", "token_refiner", "norm_out", "proj_out", "audio_proj_out",
                ],
            ),
            low_cpu_mem_usage=False,
        ),
        text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
            REPO, subfolder="text_encoder", dtype=torch.bfloat16,
            quantization_config=TransformersTorchAoConfig(
                Int8WeightOnlyConfig(version=2),
                modules_to_not_convert=[
                    "model.visual", "model.language_model.embed_tokens",
                    "model.language_model.norm", "lm_head",
                ],
            ),
        ),
    )
    pipe.load_components(workflow="t2va", dtype=torch.bfloat16)

    # version=2 int8 tensors are pinnable, which streamed offload needs, and
    # freezing removes the one autograd path quantized tensors cannot serve.
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    offload = dict(
        onload_device=torch.device(device),
        offload_device=torch.device("cpu"),
        use_stream=not args.no_stream,
    )
    pipe.transformer.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
    apply_group_offloading(pipe.text_encoder.model, offload_type="leaf_level", **offload)
    pipe.vae.to(device)
    pipe.audio_vae.to(device)
    return pipe


def main():
    args = parse_args()

    for name, val in (("width", args.width), ("height", args.height)):
        if val % 32:
            raise SystemExit(f"--{name} must be a multiple of 32, got {val}")

    frames = snap_frames(args.frames)
    if frames != args.frames:
        print(f"[probe] frames {args.frames} -> {frames} (snapped to 17n+5)")
    duration = frames / FPS
    if not 5 <= duration <= 15:
        raise SystemExit(f"duration {duration:.2f}s outside the supported 5-15s window")

    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible to torch")
    device = "cuda"
    print(f"[probe] {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).gcnArchName}), torch {torch.__version__}")
    print(f"[probe] {args.width}x{args.height}, {frames} frames ({duration:.2f}s), "
          f"{args.steps} steps, {args.precision}")

    t0 = time.perf_counter()
    pipe = build_pipeline(args, device)
    load_s = time.perf_counter() - t0
    print(f"[probe] loaded in {load_s / 60:.1f} min")

    from diffusers.utils.export_utils import encode_video

    t1 = time.perf_counter()
    results = pipe(
        prompt=args.prompt,
        num_frames=frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=torch.Generator().manual_seed(args.seed),
        output=["videos", "audio", "sampling_rate"],
    )
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = HERE / "outputs" / f"{args.tag}-{args.width}x{args.height}-{args.steps}st-{stamp}.mp4"
    encode_video(
        results["videos"][0],
        fps=FPS,
        output_path=str(out),
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )

    print(f"\n[probe] generation: {gen_s / 60:.1f} min "
          f"({gen_s / args.steps:.1f}s/step, {gen_s / duration:.0f}s per second of video)")
    print(f"[probe] load: {load_s / 60:.1f} min")
    print(f"[probe] wrote {out}")

    row = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "tag": args.tag,
        "width": args.width, "height": args.height,
        "frames": frames, "duration_s": round(duration, 2),
        "steps": args.steps, "precision": args.precision,
        "load_s": round(load_s, 1), "gen_s": round(gen_s, 1),
        "s_per_step": round(gen_s / args.steps, 1),
        "output": out.name,
    }
    with open(HERE / "timings.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
