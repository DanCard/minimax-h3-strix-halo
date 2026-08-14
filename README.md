# MiniMax-H3 on Strix Halo (gfx1151)

Notes on running [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
locally on this machine. Timings below are measured on this hardware unless a
section says otherwise; the pre-run estimates are kept at the bottom, clearly
marked, because comparing them against reality is instructive.

## Status

**Working.** MiniMax-H3 generates video with synchronized audio on a Radeon
8060S (gfx1151) via ComfyUI + ROCm 7.14.

### Measured results

All with the 4-step turbo LoRA, 124 frames (5.17s @ 24fps), joint video+audio:

| Canvas | Pixels | Tokens | Sampler s/it | Wall total | Notes |
|---|---|---|---|---|---|
| 512x288 | 147k | 17.9k | **11.2** | 96 s | models already resident |
| 384x640 | 246k | 29.8k | not recorded | 228 s | includes ~45s model reload |
| 896x512 | 459k | 55.6k | **123–126** | 571 s | step-to-step gap, steady across all 4 |
| 1344x768 | 1032k | 125k | ~550 (est.) | ~40 min (est.) | trained canvas; abandoned partway, see below |

A 22-frame (0.92s) clip at 512x288 sampled in 6.6 s.

**Resolution is the expensive axis, not steps, and the scaling looks close to
quadratic in token count.** Tokens are the latent grid, `W/16 x H/16 x 31` at
124 frames:

| Canvas | Latent grid | Tokens | vs. baseline | Sampler s/it | vs. baseline |
|---|---|---|---|---|---|
| 512x288, 124f | 32x18x31 | ~17.9k | 1.00x | 11.2 | 1.00x |
| 896x512, 124f | 56x32x31 | ~55.6k | 3.11x | ~124 | **9.93x** |

3.11² = 9.7 predicted against 9.93 measured, giving:

```
s/step ≈ 11.2 × (tokens / 17.9k)²
```

**Two caveats on that fit.** It rests on two sampler-rate points, and the
512x288 baseline is the shakier of them — at that size a larger share of any
wall-clock measurement is fixed overhead. 384x640 is deliberately excluded: only
its wall time was recorded, and wall time folds in text encode, VAE decode and
mux, so it is not comparable to a sampler rate. (Wall/4 there is ~46 s/it against
~31 predicted — the gap is overhead, not a scaling failure, but it cannot be used
as evidence either way.) Treat the formula as a good planning heuristic that
happens to fit two clean points, not as a confirmed exponent. See open thread 4.

The turbo LoRA cuts *how many* steps run; it cannot touch what each step costs.

**On the abandoned 1344x768 run.** An earlier version of this file called that a
memory cliff — GPU pegged at 100%, VRAM at 96%, still on step 1 of 4 after 16
minutes — and inferred that no memory-efficient attention backend was dispatching
on gfx1151. **That reading was wrong.** The formula above predicts ~9 min/step at
125k tokens, and a first step of >16 min including MIOpen autotune for unseen
shapes is consistent with it. Nothing fell over; the run was simply slow, and
would have finished in roughly 40 minutes. It was killed for impatience, not
failure. 768p is expensive here, not broken.

**Practical envelope:** 384x640 and below is an interactive loop (~2-4 min/clip
with models resident). 896x512 at ~9.5 min is a reasonable quality/time
compromise. Full 1344x768 is a ~40 min "start it and walk away" proposition —
slow, but a known quantity rather than a wall.

Two things are measured rather than assumed:

- torch 2.13.0+rocm7.2 on gfx1151: **30 TFLOPS** bf16 (4096^3 matmul). That is
  ~0.4x a 3090's dense bf16 throughput, better than the "3x slower" guess this
  file originally carried.
- torchao `Int8WeightOnlyConfig(version=2)` runs on gfx1151 at **0.94%**
  relative error vs bf16. int8 works here.

### Plan change: quantized weights, ComfyUI (not BF16 + diffusers)

Originally this pursued the official diffusers path: download 144GB of BF16 and
let torchao quantize to int8 on load (~75GB resident). Abandoned mid-download.

Why: the documented recipe passes `low_cpu_mem_usage=False`, so weights
materialize *before* being quantized. Loading the 66.3GB transformer, quantizing
to ~33GB, then loading the 66.7GB conditioner on top plausibly peaks near 100GB
against 124GB of RAM. Tight, and pointless when pre-quantized weights exist.

Now using community pre-quantized weights (~44GB, no load-time spike) driven by
ComfyUI. `run_probe.py` targets the diffusers path and is kept for reference but
is **not** the active route.

## The machine

| | |
|---|---|
| GPU | AMD Radeon 8060S (Ryzen AI Max+ 395), `gfx1151`, 40 CU RDNA3.5 |
| ROCm | 7.14.0 (latest release, 2026-07-15; officially supports gfx1151) |
| RAM | 124 GB unified, ~120 GB addressable by the GPU via GTT |
| Disk | 1.1 TB free, single filesystem (`/dev/nvme0n1p2`) |
| Torch | `2.12.0+rocm7.2` in `~/.venvs/transcribe` — verified working on gfx1151 |

The 512 MB "VRAM" reported by `rocm-smi` is just the BIOS carve-out. GTT is the
real pool and 120 GB of it is available, which is the correct config here.

## What H3 is

Not a text LLM. An omni-modal **video + audio generator** — text/image/video in,
video with synchronized stereo audio out. One transformer denoises a single
packed sequence holding text conditioning, video latents and audio latents
together. No separate vocoder.

- 33B Omni-Transformer + Qwen3-VL-32B as conditioner (reads hidden state after
  decoder layer 50, LM head unused)
- 24 fps, 5–15 s, 768 px short edge, 32 kHz stereo audio
- Released BF16. Transformer 61.7 GB, conditioner 62.1 GB

## Hard constraints (verified from the diffusers integration)

These are not API validation you can bypass — they're enforced in the pipeline
blocks and the VAE geometry.

1. **Minimum 124 frames (~5.17 s) — in diffusers only.** `num_frames` snaps up
   to the next `17n + 5` the video VAE can decode, and the diffusers pipeline
   additionally requires the resulting duration to land in 5–15 s, making
   `17×7+5 = 124` the floor there. Every official example uses `num_frames=124`.

   **ComfyUI does not enforce the 5-second floor.**
   `EmptyMiniMaxH3LatentAV` takes `length` down to 5 on the same `17n+5` grid,
   and a 22-frame (0.92 s) clip renders fine — measured above at 6.6 s of
   sampling. Sub-124-frame lengths are off-distribution (the model was trained at
   ~124–362) so quality suffers, but they run, which makes them the cheapest way
   to time a configuration.
2. **No CFG lever.** The released checkpoints are guidance-distilled — guidance
   is baked into the weights. There is no `guidance_scale`, no `negative_prompt`,
   no guider, and every step is exactly one forward pass. The usual 2× win from
   disabling CFG is already spent, and published benchmarks already include it.
3. **No 2K locally.** H3-Regenerate-2K is not open-sourced. Local means 768p base.
4. **Canvas must be multiples of 32.** Trained canvas is 1344×768
   (= 1,032,192 px, the pipeline's `canvas_max_pixels`).

## Speed levers that do exist

| Lever | Effect | Notes |
|---|---|---|
| Canvas size | **~Quadratic in tokens** — measured 9.93× cost for 3.11× tokens | By far the biggest lever |
| `num_inference_steps` | Linear | Exposed in diffusers; *not* in the HTTP API. Turbo LoRA already puts this at 4 |
| Duration | Linear-ish in frames, which are linear in tokens | 5.17 s is already the fast end of the diffusers range; ComfyUI goes lower |

Note the measured quadratic disagrees with MiniMax's own documentation, which
reports 960×544 as ~2.3× faster per step than 1344×768. Those canvases differ by
1.98× in tokens, so quadratic scaling predicts ~3.9×. The likely explanation is
that the reference deployment runs a proper fused-attention kernel where the
quadratic term is far better amortized; on the pytorch attention path here it is
not. **Don't port MiniMax's canvas/time ratios to this hardware — they are
optimistic.**

## Pre-run estimates (superseded — kept for calibration)

Everything in this section was written *before* anything ran. It is preserved
because the errors are large and instructive; see "How the estimates did" at the
end of it. **Use the measured table at the top of this file, not these numbers.**

Published H3 benchmarks on discrete NVIDIA cards:

| GPU | Clip | Time |
|---|---|---|
| RTX 4090 | 5 s | 3–6 min |
| RTX 4090 | 15 s @ 0.6 MP | ~15 min |
| RTX 3090 | 5 s | 7–51 min (resolution-dependent) |
| RTX 3090 | 10 s | ~12 min |
| RTX 3090 | 15 s | ~25 min |

Strix Halo lands roughly in RTX 3060 territory for diffusion video work even
with ROCm-optimized nodes — call it ~3× slower than a 3090. That gives:

- **Full 768p, ~5 s clip: ~20–40 min** (usable quality, painful iteration)
- **Minimal probe** (124 frames, 512×288, reduced steps): **~5–15 min** of
  denoising, plus load time which may dominate

For context, MiniMax's own reference deployment is `--num-gpus 4
--ulysses-degree 4`. Single-consumer-GPU use is against the grain of the target.

**The honest read:** even the good case is ~20–40 min per attempt at usable
resolution, and video prompting normally takes several attempts. This is a
"start it before bed" tool, not an iteration loop. Comparable models on this
exact hardware — LTX-2 does 10 s with audio in ~10 min — are far more livable.

### How the estimates did

| Claim | Predicted | Actual | |
|---|---|---|---|
| Minimal probe, 512x288 | 5–15 min | **96 s** | 3–9× too pessimistic |
| Full 768p, 5 s clip | 20–40 min | ~40 min (est., never completed) | roughly right |
| Canvas scaling | "6–8× faster than trained canvas" at 512x288 | **~49×** | badly wrong |

The pattern: small canvases were far *better* than predicted and large ones
about as predicted, because the estimate scaled on **pixel count** while the real
cost scales on **tokens squared**. Any single-point extrapolation from a published
benchmark inherits that error. The 96 s result also shows the "not an iteration
loop" conclusion was wrong at small canvas — it is a perfectly comfortable loop
below ~30k tokens.

## Weights actually in use

Fetched by `fetch_quant.py` into the default HF cache (~44GB):

| File | Size | Notes |
|---|---|---|
| `MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors` | 21.0 GB | Transformer, int8, highest-quality pruned variant |
| `qwen3vl_32b_minimax_h3_int4_convrot.safetensors` | 15.0 GB | Text encoder. The repo guide recommends int8 but only int4/nvfp4 exist |
| `minimax_h3_video_vae_fp16.safetensors` | 5.2 GB | |
| `minimax_h3_audio_vae_fp32.safetensors` | 0.6 GB | |
| `minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` | 2.0 GB | **4-step turbo LoRA** |

From `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` and `lightx2v/Minimax-h3-Turbo`.
FL2VA covers t2v and i2v; Ref2VA (omni-references) not fetched.

These are **pruned community conversions**, not official weights — quality is
unverified against the full model. That is an accepted tradeoff for a speed probe.

**The turbo LoRA is the biggest lever on generation time found so far.** It is a
step-distillation LoRA that cuts denoising to 4 steps. Since the base checkpoint
is already guidance-distilled (one forward pass per step), 4 steps is 4 forward
passes — far cheaper than anything achievable by shrinking the canvas alone.

## Memory plan

BF16 does not fit: transformer 61.7 GB + conditioner 62.1 GB = 123.8 GB against
124 GB of RAM. **int8 is required**, not optional.

The documented consumer recipe quantizes both large components to int8 with
torchao and streams transformer blocks from host RAM, expecting **~75 GB of host
RAM**. Against 124 GB that is a comfortable fit — better than the 12–16 GB cards
the recipe was written for, since block-level offload won't thrash nearly as hard.

Community quants (GGUF Q2–Q5, pruned INT4/INT8, NVFP4) exist at 8.9–21.6 GB, but
they're third-party conversions with unverified loaders and quality. Reasonable
place to start given the size difference; fall back to the int8 diffusers recipe
if output looks degraded.

**Do not** `hf download` the bare repo — it is 498 GB of largely duplicate
variants. Scope it:

```bash
hf download MiniMaxAI/MiniMax-H3 --include "model_index.json" "FL2VA/*"
```

Or let diffusers fetch per-workflow, which pulls only what the workflow needs.

## gfx1151-specific notes

- **`HSA_USE_SVM=0` is important.** Disabling shared virtual memory management
  prevents SVM thrashing on unified memory; it was worth a ~3× speedup on
  Hunyuan Video on this hardware.
- ROCm 7.14 is *newer* than the 7.10/7.11 builds where the gfx1151 video-gen
  fixes landed, so the stack is in good shape.
- `_flash_3_hub` attention backend is Hopper-only — not available here.
- Working Strix Halo ComfyUI toolboxes exist:
  https://github.com/kyuz0/amd-strix-halo-comfyui-toolboxes

## Layout

```
~/projects/minimax-h3/     this repo — scripts, notes
~/.venvs/minimax-h3/       venv (matches the ~/.venvs/transcribe precedent)
~/.cache/huggingface/      weights (default HF cache; HF_HOME is unset)
```

Weights deliberately stay in the HF cache: content-addressed dedupe across
tools, and no risk of git-adding 20 GB.

## Reproducing

Weights and ComfyUI are both gitignored; these scripts fetch and repair them.

```bash
# 1. venv with ROCm torch (Python 3.13; 3.14 wheels exist but are untested here)
uv venv --python 3.13 ~/.venvs/minimax-h3
uv pip install --python ~/.venvs/minimax-h3/bin/python \
  --index-url https://download.pytorch.org/whl/rocm7.2 \
  torch==2.13.0+rocm7.2 torchvision torchaudio

# 2. ComfyUI, minus its torch pins (they would clobber the ROCm build)
git clone https://github.com/comfyanonymous/ComfyUI.git
grep -vE '^(torch|torchvision|torchaudio)\s*$' ComfyUI/requirements.txt > /tmp/r.txt
uv pip install --python ~/.venvs/minimax-h3/bin/python -r /tmp/r.txt

# 3. weights (~44GB) and the repair pass
~/.venvs/minimax-h3/bin/python fetch_quant.py
~/.venvs/minimax-h3/bin/python repair_weights.py

# 4. symlink models_fixed/* into ComfyUI/models/{diffusion_models,text_encoders,vae,loras}

# 5. serve -- HSA_USE_SVM=0 matters on unified memory
cd ComfyUI && HSA_USE_SVM=0 python main.py --listen 127.0.0.1 --port 8188

# 6. generate
#    fast iteration (~96 s):
python probe_comfy.py --width 512 --height 288 --length 124 --steps 4 \
  --seed 7 --tag quick --prompt "..."
#    quality/time compromise (~9.5 min):
python probe_comfy.py --width 896 --height 512 --length 124 --steps 4 \
  --seed 7 --tag demo --prompt "..."
```

Width and height must be multiples of 32. Every run appends a row to
`timings.jsonl`.

### The weights are corrupt as published

Every file in `Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot` has an upload-tool
marker appended past the end of its declared tensor data:

```
\nL2P_bypass_minimax_h3_audio_vae_fp32.safetensors_1785752000\n
```

safetensors rejects this outright with *"Error while deserializing header:
incomplete metadata, file not fully covered"*, so ComfyUI cannot load any of
them. Trailing byte counts seen: 72, 85, 61, 61.

`repair_weights.py` truncates each file to its declared end. This is verified,
not assumed: `Comfy-Org/MiniMax-H3` publishes a clean copy of the audio VAE, and
the truncated Abiray file is **sha256-identical** to it.

`Comfy-Org/MiniMax-H3` is the authoritative ComfyUI repackaging and ships the
same files uncorrupted (plus an int8 text encoder Abiray's README recommends but
does not actually contain). Prefer it if you are starting fresh.

## Open threads

1. **Attention backend.** ComfyUI logs "Using pytorch attention", and the
   measured scaling is almost perfectly quadratic — which is what an *unfused*
   attention path looks like. This is no longer needed to explain 768p (the token
   math alone does that), but it is the most promising remaining speedup: a fused
   kernel would attack the quadratic term directly, which is exactly where the
   cost is. Worth trying ComfyUI's alternate cross-attention flags and measuring
   the exponent again.
2. **Unpruned weights.** Everything measured here uses *pruned* community
   conversions, so output quality does not represent official H3.
   `Comfy-Org/MiniMax-H3` ships unpruned bf16 (66GB) and int8 (34GB) variants.
3. **Finish a 1344x768 run.** Now predicted at ~40 min rather than "impossible".
   Worth doing once to confirm the quadratic model holds at 125k tokens and to
   see the trained canvas at its intended resolution.
4. **Pin down the exponent properly.** The quadratic fit rests on two
   sampler-rate points. `probe_comfy.py` records wall time only; parsing the
   tqdm s/it out of the ComfyUI log into `timings.jsonl` would make every run
   contribute a comparable number, and a third and fourth canvas would show
   whether the exponent is really 2.0 or just near it.

`run_probe.py` targets the diffusers path, which was abandoned before it ever
ran. It is kept for reference only and has never been executed.

If local video generation is the goal rather than H3 specifically, **LTX-2 or
Wan 2.2** have prepackaged Strix Halo toolboxes and are likely a gentler start.

## Sources

- [Model card](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- [diffusers pipeline docs](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3) — the authority on frame/canvas constraints
- [ComfyUI day-0 support](https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui)
- [Community quants](https://comfyui-wiki.com/en/news/2026-08-03-minimax-h3-community-quants)
- [LTX-2 on gfx1151](https://github.com/ROCm/TheRock/discussions/2845)
- [Hunyuan `HSA_USE_SVM=0` fix](https://github.com/ROCm/TheRock/discussions/2684)
