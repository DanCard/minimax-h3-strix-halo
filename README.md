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
| 384x640 | 246k | 29.8k | **21.0** | 228 s | s/it from a later fresh-server run |
| 896x512 | 459k | 55.6k | **123–126** | 571 s | step-to-step gap, steady across all 4 |
| 512x896 | 459k | 55.6k | **128 → 176** | 12 min 26 s | portrait; drifted upward, aged server |
| 1344x768 | 1032k | 125k | **766–858** | **56 min 47 s** | trained canvas, completed |
| 768x1344 | 1032k | 125k | **811–823** | **58 min 21 s** | trained canvas in portrait; flat |

A 22-frame (0.92s) clip at 512x288 sampled in 6.6 s.

**The trained canvas works, in either orientation.** 1344x768 and 768x1344 are
not out of reach on this hardware — each is just a ~1 hour job for a 5.17 s clip.

**Transposing a canvas is free.** 512x896 and 896x512 have identical token
counts, as do 768x1344 and 1344x768 — attention cost depends only on sequence
length. Confirmed twice: 512x896 opened at 128 s/it against 896x512's measured
123–126, and 768x1344 averaged 819 s/it against 1344x768's ~814, a 0.6% gap.
A measured landscape run is therefore a reliable predictor of its portrait twin,
which is the single most useful estimating trick in this file.

**Resolution is the expensive axis, not steps, and it scales worse than
quadratically.** Tokens are the latent grid, `W/16 x H/16 x 31` at 124 frames:

| Canvas | Latent grid | Tokens | vs. baseline | Sampler s/it | vs. baseline | Pairwise exponent |
|---|---|---|---|---|---|---|
| 512x288, 124f | 32x18x31 | ~17.9k | 1.00x | 11.2 | 1.00x | — |
| 384x640, 124f | 24x40x31 | ~29.8k | 1.66x | 21.0 | 1.9x | **1.24** |
| 896x512, 124f | 56x32x31 | ~55.6k | 3.10x | ~124 | 11.1x | **2.85** |
| 768x1344, 124f | 48x84x31 | ~125k | 6.98x | ~819 | 73.1x | **2.33** |

**There is no single exponent.** An earlier version of this file fitted
`^2.2` across three points and reported the pairwise exponents as "drifting
upward" (2.27 then 2.32). Adding a fourth point breaks that story outright: the
pairwise exponents are **1.24, 2.85, 2.33** — non-monotonic, spanning more than
a factor of two. The tidy power law was an artefact of having too few points.

The honest summary:

- **At the top end, ~2.3 holds and extrapolates well.** It is the only region
  with enough signal to trust for planning.
- **At the low end the measurement noise swamps the effect.** The same 384x640
  canvas measured 21.0 s/it on a fresh server and 27.8 s/it on an aged one — a
  32% spread, larger than the gap between neighbouring canvases. Sub-30k-token
  points cannot resolve an exponent at all.
- **Every timing in this file carries server age as an unrecorded confound**
  (see *Server age, not run length* below). Numbers taken on a fresh server and
  a tired one are not comparable, and earlier runs did not track which.

Practical consequence: **anchor to the nearest measured point, and prefer a
transpose anchor over any exponent.** Extrapolation across more than ~2x in
tokens is guesswork.

The turbo LoRA cuts *how many* steps run; it cannot touch what each step costs.

**On the earlier "memory cliff".** A previous version of this file described an
abandoned 1344x768 attempt as a memory cliff, and inferred that no
memory-efficient attention backend was dispatching on gfx1151. **Both claims were
wrong.** The completed run settles it — during it:

- swap traffic (`si`/`so`) was flat zero
- VRAM sat at **89%**, *lower* than the 896x512 run that finished fine
- the GPU was pegged at 100% the whole time
- per-step times got **faster** as it progressed (858 → 848 → 784 → 766 s),
  the opposite of a system under growing pressure

Nothing degrades and nothing falls over. The original attempt was killed for
impatience, not failure. 768p is expensive here, not broken.

**Server age, not run length, is what slows a run down.** Per-step times drift
upward *within* a run, but only on a server that has already done work:

| Run | Server state | Per-step |
|---|---|---|
| 512x896, 4 steps | ~5 runs deep, 94% VRAM | 128 → **176 s** (+38%) |
| 384x640, 20 steps | 2 runs deep | 27.8 s, flat across all 20 |
| 768x1344, 4 steps | **freshly restarted** | 812.7 / 810.8 / 813.2 / 823.1 s, flat |

The 768x1344 run is the decisive one: three times longer and 2.25x the tokens of
the run that drifted 38%, yet dead flat for 55 minutes. So the drift is **not** a
function of run length, canvas size or thermals (the GPU idles at 36 °C) — it
tracks how much work the *process* has already done, i.e. allocator
fragmentation of the GTT/VRAM carve-out.

**Restart ComfyUI before any long or timed run.** It costs ~45 s and it is the
difference between a predictable number and a lottery. It is also a precondition
for any timing worth recording.

**Practical envelope:**

| Canvas | Time | Use |
|---|---|---|
| 512x288 | ~1.5 min | prompt iteration |
| 384x640 | ~4 min | cheap A/B tests |
| 896x512 / 512x896 | ~9.5 min | quality/time sweet spot — 6x cheaper than 768p |
| 1344x768 / 768x1344 | ~58 min | final render, start it and walk away |

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
- **Lip-synced speech in 11 languages**, generated jointly with the picture —
  verified working here, see *Lip-synced speech works* below
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
| Canvas size | **~Tokens^2.3 at the top end** — measured 73× cost for 7.0× tokens | By far the biggest lever, and the only one that improved quality |
| `num_inference_steps` | Linear | Exposed in diffusers; *not* in the HTTP API. Turbo LoRA already puts this at 4, and raising it made output *worse* |
| Duration | Linear in tokens, same as pixels | Frames and pixels are the same currency: 768x1344 at 15 s is ~363k tokens, a ~10 hour job |

Note the measured scaling disagrees sharply with MiniMax's own documentation,
which reports 960×544 as ~2.3× faster per step than 1344×768. Those canvases
differ by 1.98× in tokens; the top-end exponent measured here predicts ~4.5×.

Real per-step cost is `a·N + b·N^k` — projections and MLPs are linear, attention
superlinear. On a deployment where `b` is small (fused kernels, dedicated HBM),
the linear term still matters at these sizes and the observed ratio looks close
to linear. Here `b` is large enough that the superlinear term runs the show by
~56k tokens, and `k` itself appears to grow with N. **Don't port MiniMax's
canvas/time ratios to this hardware — they are optimistic by roughly 2×.**

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
| Full 768p, 5 s clip | 20–40 min | **57 min** | 1.4–2.9× too optimistic |
| Canvas scaling | "6–8× faster than trained canvas" at 512x288 | **~73×** | badly wrong |

The pattern: small canvases came in far *better* than predicted and the large one
*worse*, because the estimate scaled on **pixel count** while real cost scales on
tokens to the ~2.2. That error compounds in both directions from wherever the
anchor benchmark sat. Any single-point extrapolation inherits it.

The 96 s result also shows the "not an iteration loop" conclusion was wrong at
small canvas — it is a perfectly comfortable loop below ~30k tokens. The "start
it before bed" framing turns out to be right only for the trained canvas.

The in-flight estimates were not much better. A running tally:

| In-flight estimate | Predicted | Actual | |
|---|---|---|---|
| 1344x768, from a 2-point quadratic fit | ~40 min | **56:47** | 1.4× optimistic |
| 512x896, from the 896x512 transpose | 9.5–10 min | **12:26** | 1.3× optimistic |
| 8 steps at 384x640, from a stale baseline | ~7 min | **3:39** | 1.9× pessimistic |
| **768x1344, from the 1344x768 transpose** | **~55 min** | **58:21** | **1.06×** |

Two lessons, and they point the same way. The 512x896 miss was not the transpose
anchor's fault — the sampler rate it predicted was right — but the VAE
decode/mux overhead was 133 s, not the 30–45 s carried over from far smaller
canvases; **overhead scales with pixels x frames too.** The 8-step miss came
from anchoring to a baseline measured on an aged server.

The one estimate that landed used a **same-token-count transpose anchor on a
freshly restarted server**, with overhead taken from a comparable canvas. That
is the recipe: **measure the canvas you intend to use, control the server state,
and never extrapolate more than ~2x in tokens.**

## Weights actually in use

Fetched by `fetch_quant.py` into the default HF cache (~44GB):

| File | Size | Notes |
|---|---|---|
| `MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors` | 21.0 GB | Transformer, int8, highest-quality pruned variant |
| `qwen3vl_32b_minimax_h3_int4_convrot.safetensors` | 15.0 GB | Text encoder. Abiray's guide recommends int8 but only ships int4/nvfp4; `Comfy-Org/MiniMax-H3` does have int8 (27GB), tested and indistinguishable — int4 is the better trade |
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
- **MIOpen `Conv3d` segfaults above a size threshold**, which breaks all image
  conditioning until patched — see *Image conditioning segfaults* below.
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

# 5. patch the vision-tower Conv3d -- required for any image conditioning,
#    see "Image conditioning segfaults" below
#    ComfyUI/comfy/text_encoders/qwen35.py, Qwen35VisionPatchEmbed.forward

# 6. serve -- HSA_USE_SVM=0 matters on unified memory
cd ComfyUI && HSA_USE_SVM=0 python main.py --listen 127.0.0.1 --port 8188

# 7. generate
#    fast iteration (~96 s):
python probe_comfy.py --width 512 --height 288 --length 124 --steps 4 \
  --seed 7 --tag quick --prompt "..."
#    quality/time compromise (~9.5 min):
python probe_comfy.py --width 896 --height 512 --length 124 --steps 4 \
  --seed 7 --tag demo --prompt "..."
#    image-to-video (needs the step 5 patch):
python probe_comfy.py --width 448 --height 704 --length 124 --steps 4 \
  --seed 7 --tag anim --image photo.jpg --outdir run1 --prompt "..."
```

Width and height must be multiples of 32. Every run appends a row to
`timings.jsonl`. `--image`/`--last-image` add first/last-frame conditioning;
each is scaled to the canvas with a centre crop, because the node itself
plain-stretches whatever it receives. `--outdir` saves into a subdirectory of
`ComfyUI/output/`. `--unet` and `--clip` switch the diffusion model and text
encoder by filename, for A/B tests; both are recorded in `timings.jsonl`
alongside the scheduler and LoRA strength, without which ablation rows are
indistinguishable afterwards.

Defaults worth knowing: **`sgm_uniform`** scheduler and **`--lora-strength
0.85`**, both chosen by sweep (see below), `res_multistep` sampler, 4 steps.
Raising the step count at full LoRA strength makes output *worse*, not better;
at 0.85 more steps help slightly but cost proportionally. **Exception: use
`--steps 8` for anything with speech** — audio and video converge at different
rates, see *Lip-synced speech works* below.

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

### Lip-synced speech works, and it is free

H3 generates speech in the same pass as the picture — it is not an add-on. The
prompts in this file mostly asked for ambience (`Audio: waves washing onto
sand`), which is why speech never appeared. It needs a specific syntax: stable
speaker IDs outside the tag, and only the language plus the spoken words inside
it.

```
She looks directly at the camera and speaks with a clear, confident voice (S1):
<d>[English] The atmosphere is breathable. We can proceed.</d>
```

Verified objectively rather than by ear — Whisper was run over the *generated*
audio, knowing nothing about the prompt:

```
[00:01.390 --> 00:04.790] SPEAKER_00: The atmosphere is breathable. We can proceed.
```

Verbatim, at 32kHz stereo, with diarization reporting 1 speaker across 2 turns
and silence at the head and tail rather than talking through the whole clip.

It survives the turbo LoRA at 4 steps, which was the real risk — lip-sync is
fine temporal structure and aggressive distillation should have destroyed it.
Cost was 232 s at 448x704, the same as any other clip that size. Speech rides
along for free.

**But use 8 steps for speech. This is the one default that forks by use case.**

| Use case | Steps | Why |
|---|---|---|
| Video only | **4** | 8 steps measurably *reduced* image detail (−18%) |
| **Dialogue / speech** | **8** | 4-step audio has an audible ringing artefact; 8 is clearly cleaner |

The 4 → 8 improvement is **very noticeable** by ear. 16 is slightly better again
but well into diminishing returns and not worth 2x the cost over 8.

Video and audio are denoised in the same packed sequence yet evidently converge
at **different rates**, and the turbo LoRA is distilled against the *video*
objective — so no single step count serves both. That is the whole explanation
for the split.

**Intelligibility is not quality**, and conflating them cost a wrong call here.
An earlier version of this section concluded 4 steps was fine for dialogue
because Whisper transcribed it perfectly. It does, and it still does — the words
are correct at 4 steps. They are also correct at 8. What differs is the ringing
underneath, which transcription is completely blind to.

Notes for going further:

- **Keep lines short.** Sync is documented to drift on longer clips and dense
  lines, and 5.17 s fits roughly one short line per speaker.
- **Two-way dialogue needs duration, which is the expensive axis.** Real
  alternation wants 10–15 s: 362 frames at 384x640 is ~86k tokens (~25 min), at
  448x704 ~111k (~43 min).
- **Two *visible* characters want `ref2va`**, a different checkpoint
  (`minimax_h3_ref2va_pruned_int8_convrot`, 21GB) driven by portrait stills.
- **Two speakers work, even at 512x288 and 5.17 s.** A two-shot with `(S1)` and
  `(S2)` produced both lines word-perfect *in audibly distinct voices* — a clear
  female and a low gravelly male, as prompted. 148 s at 8 steps.
- **Do not trust diarization to verify this.** pyannote reported "1 speaker
  across 3 turns" for that clip, and it was wrong — the voices are obviously
  different to a listener. At ~2 s of speech per speaker there is not enough
  audio to build usable embeddings. Transcription is reliable for *words*;
  speaker separation at this clip length is not, and it says nothing about audio
  quality either. Listen instead.
- **The timeline syntax appears to be ignored at this length.** An explicit
  `At 00:02.500` for the second speaker produced continuous speech from 0.0 to
  4.67 s with no gap, unlike the single-speaker clip which had clean silence at
  head and tail. The model packed both lines in rather than honouring the
  timing. Untested at longer durations, where it may well behave.
- **Iterate at 512x288.** Audio came out intelligible and clean at the smallest
  canvas, transcribing identically, at 87 s for 4 steps (~170 s at 8). Script,
  timing and voice work can all be tuned there before committing to a large
  canvas. One seed only, so treat as promising rather than established.
- **Audio artefacts are not the container.** The muxed AAC is 131 kbps at 32kHz
  stereo with clean response to 15 kHz and no spectral holes; anything audibly
  wrong is in the generated audio, not the encode.

### Quality levers: only resolution worked

Five levers were tested against a fixed control, changing exactly one variable
each time. Detail is mean Laplacian variance over sampled frames, all clips
rescaled to a common display size so that real extra detail shows up as extra
variance rather than as a larger pixel count — but see the warning below about
how far that number can be trusted.

| Change | Detail | vs. control | Verdict |
|---|---|---|---|
| 4 steps, turbo LoRA (control) | 307.1 | — | — |
| 8 steps, turbo LoRA | 251.6 | −18% | slightly worse |
| **20 steps, no LoRA** | 85.5 | **−72%** | **much worse** |
| Unpruned int8 diffusion (34GB vs 21GB) | 278.1 | −9.5% | indistinguishable |
| **bf16 diffusion, unquantized (38GB vs 21GB)** | — | — | **indistinguishable** |
| **int8 text encoder (27GB vs int4 15GB)** | — | — | **indistinguishable** |
| Scheduler (6 swept) | — | — | small win — `sgm_uniform` |
| **768x1344 vs 384x640** | — | **9.1x** | **decisive** |

**Every component swap came back negative.** Only resolution moved the needle,
and the scheduler was worth a small free improvement. The generation-side search
is effectively exhausted: each component is either already at its best setting
or demonstrably irrelevant.

- **Step count alone is not the lever — but it interacts with LoRA strength.**
  Dropping the turbo LoRA and running 20 steps produced 3.6x *less* detail,
  corroborated independently by file size (392KB vs 695KB — H.264 encodes soft
  video small). 8 steps was mildly worse than 4. The model is guidance-distilled
  at the base, so removing the LoRA does not restore CFG; it just samples a
  schedule nothing was tuned for.

  That test held `lora_strength` at 1.0 throughout and concluded "steps are not
  the lever", which was true only *at full strength*. Sweeping strength
  separately shows the two axes are not independent:

  | Strength @ steps | Wall | Read |
  |---|---|---|
  | 1.0 @ 4 (old default) | 349 s | baseline; softer, flatter eyes |
  | **0.85 @ 4 (now default)** | **313 s** | **better eye definition and lashes at the same cost** |
  | 0.85 @ 6 | 629 s | best of the four — finest hair, most defined eyes |
  | 0.70 @ 8 | 393 s | comparable to 0.85 @ 6; extra steps buy nothing |

  **`--lora-strength 0.85` at 4 steps is free quality.** Use 6 steps only for a
  final render where ~2x sampling is acceptable — at 768x1344 that is ~2 hours.
  These four are much closer together than the scheduler or resolution
  differences and sit near the edge of what one seed supports: the 0.85-over-1.0
  gap looks real, the 0.85@6-vs-0.70@8 gap is not defensible.
- **Pruning is not the lever.** Unpruned int8 landed within noise of pruned.
  One seed only, so this rules out a *large* effect, not a small one.
- **Quantizing the diffusion model is not the lever either.** This needed a
  second test: pruned int8 vs *unpruned* int8 varies **pruning**, not
  precision — both arms are int8 — so the first A/B never touched quantization
  at all. An earlier version of this file claimed every quantized component had
  been checked against a higher-precision version, which was not true of the
  component doing the actual generating. Pruned **bf16** (38GB, unquantized)
  against pruned int8 (21GB), pruning held constant, settles it: no visible
  difference, and bf16's file is *smaller* (631KB vs 704KB), which argues
  against it resolving more detail.

**int8 is not the faster option.** Matched conditions — 448x704, first run on a
freshly restarted server, int4 encoder, only the diffusion weights differing:

| Diffusion weights | Sampler |
|---|---|
| pruned int8, 21GB | 59.8 s/it |
| **pruned bf16, 38GB** | **49.6 s/it** |

bf16 is **~17% faster** at 1.8x the weights, so dequantization costs more here
than the memory bandwidth it saves — int8 has no fast path on gfx1151 worth the
overhead. An earlier reading of this comparison had it backwards by measuring
bf16's cold run against an int8 run that was already autotuned; per-step numbers
are only comparable at matching server age *and* matching autotune state.

So int8's benefit is **footprint, not speed**. That still matters at the trained
canvas: 1344x768 with int8 already sits at 89% VRAM, and bf16 adds 17GB on top,
so int8 (or a fresh measurement) is the safer choice there. Below that, bf16 is
simply better — same quality, faster, and no quantization to reason about.
- **Quantizing the text encoder to int4 costs nothing measurable.** int8
  (27GB) against the int4 (15GB) in use, at 448x704 with `sgm_uniform`: both
  rendered the same prompt detail — including a "bright blue eyes" term added
  specifically to give a fine attribute to lose — with comparable rendering and
  no consistent winner. This was the most suspicious component left, on the
  reasoning that int4 on a 32B model is where damage usually shows. It is not:
  a text prompt is a short, low-bandwidth conditioning signal, so there is
  little for the extra precision to preserve. **Stay on int4** — 12GB smaller
  and faster to load, for no measurable cost. The encoder runs once per
  generation, not per step, so it cannot affect the sampling rate either way.
- **Resolution is the lever.** 9.1x the fine detail from 384x640 to 768x1344,
  and 2.85x at the 512x896 waypoint. Nothing else came close.

What remains is no longer parameter tuning: post-processing (upscale plus frame
interpolation, cheap but cosmetic) or `first_frame` conditioning anchored to a
dedicated image model, which is the real remaining ceiling — image models are
far better per unit of compute at a single frame than a video model is, so H3
would only have to move a good subject rather than invent one.

### Scheduler: sgm_uniform, and a warning about measuring this

Six schedulers at 448x704, seed 42, `res_multistep`, 4 steps, one variable:

| Scheduler | Verdict |
|---|---|
| **`sgm_uniform`** | **Best — now the default.** Natural tone, slightly better hair separation and cleaner background than `simple` |
| `simple` | Clean; the previous default |
| `beta` | Harsh — crushed shadows, blown highlights, waxy skin |
| `kl_optimal` | Overcooked — oversaturated and painterly |
| `linear_quadratic` | **Broken** — posterized and solarized |
| `normal` | **Broken** — blocky coloured noise |

`karras` and `exponential` were skipped deliberately: both shape sigmas for the
variance-exploding formulation and H3 is flow-matching.

The win is small but free — same step count, same cost, better default forever.

**The detail metric failed completely here, and that is the more useful
finding.** Ranked by mean Laplacian variance, the top two results were
`linear_quadratic` (5.41x the control) and `normal` (4.96x) — *the two broken
ones*. The metric counted posterization and colour noise as detail. Two distinct
traps, both worth knowing before trusting any automated image score:

- **Contrast masquerades as detail.** `beta` scored 1.69x purely by having
  higher contrast (61.3 vs 55.1 in the same table). Crushed blacks and blown
  highlights raise high-frequency energy without resolving anything.
- **A different schedule produces a different *composition*.** Changing the
  scheduler changes the denoising trajectory, so the camera push-in lands
  elsewhere and the subject's apparent size changes. Judged as thumbnails, the
  more zoomed-in variant looks sharper. That is framing, not fidelity.

A frame-to-frame difference column was the only automated signal that caught it:
the broken runs showed ~3x the motion of the good ones, which is noise, not
movement.

**Use it only for gross differences.** It is trustworthy at the 3.6x and 9.1x
gaps above and actively misleading below roughly 2x. For anything subtle, pull
native-resolution crops and look — a labelled contact sheet takes about twenty
seconds and would have prevented the wrong call here.

**This generalises, and audio is worse.** Every scalar proxy reached for during
this work failed on the subtle comparisons while working fine on the gross ones:

| Metric | Failure |
|---|---|
| Laplacian variance | ranked the two *broken* schedulers 1st and 2nd |
| Laplacian variance | read `beta`'s crushed blacks and blown highlights as detail |
| Harmonic-to-noise ratio | rewarded the ringing artefact — it *is* peak-to-valley ratio, and a ring is a peak |
| Tonal prominence | picked out 90 Hz rumble, then pointed the wrong way in the 1–10 kHz band |
| Whisper transcription | perfect at 4 and 8 steps alike; blind to the artefact entirely |
| pyannote diarization | reported 1 speaker for a clip with two audibly distinct voices |

Three separate audio measurements disagreed with a listener who could hear the
difference immediately, and the listener was right. A fourth — diarization —
was then proposed *in this file* as the objective check for multi-speaker
output, and failed on its first real use for the same reason: too little signal
for the tool's assumptions. The rule that survived all
of it: **use metrics to catch things that are grossly broken, and eyes and ears
for everything else.** Where a proxy and a human disagree on a subtle call, the
human wins — the proxy is usually measuring the artefact.

### Image conditioning segfaults: a MIOpen Conv3d bug

Any image fed to the Qwen3-VL conditioner — `first_frame`/`last_frame` on
`MiniMaxH3ImageToVideo`, or `ref_images` — killed the server. Three attempts
produced three *different* failures: a 732 s stall at 1% GPU, a
`Fatal Python error: Segmentation fault` inside `load_torch_file`, and a
`malloc(): unaligned tcache chunk detected` inside
`qwen35.py:fast_pos_embed_interpolate`. Three unrelated crash sites is the
signature of heap corruption rather than a logic bug: the abort lands wherever
the next allocation happens to touch the damaged heap, which is why it looked
like three separate problems.

It is none of the things it looked like. The vision tower is intact (351
`visual.*` tensors, BF16, `pos_embed` [2304, 1152] = the correct 48x48 grid, no
`L2P_bypass` markers), `grid_thw` is a sane `[[1, 44, 28]]`, and running with
`--disable-async-offload --disable-pinned-memory` reproduces the abort at the
same line. The cause is the vision tower's patch embedding — a `Conv3d`.

Minimal reproduction, no ComfyUI and no H3 weights:

```python
conv = nn.Conv3d(3, 1152, [2, 16, 16], stride=[2, 16, 16]).cuda()
x = torch.randn(1232, 3, 2, 16, 16, device="cuda")
y = conv(x)          # SIGSEGV, core dumped
```

The trigger is a size threshold on `batch x out_channels`, not a shape quirk:

| batch | out_channels | result |
| --- | --- | --- |
| 64 | 1152 | ok |
| 256 | 1152 | **SIGSEGV** |
| 1232 | 16 | ok |
| 1232 | 128 | **SIGSEGV** |

Kernel size is irrelevant (crashes at [2,2,2] through [2,16,16]) and fp32, bf16
and fp16 all crash. MIOpen appears to switch algorithm above a workspace
threshold, onto a path that is broken on gfx1151.

**The fix** exploits the fact that this convolution has `stride == kernel_size`
and receives input pre-shaped to exactly one patch per row — so it is a per-row
matmul wearing a convolution costume:

```python
# comfy/text_encoders/qwen35.py, Qwen35VisionPatchEmbed.forward
weight, bias = comfy.ops.cast_bias_weight(self.proj, x)
return F.linear(x.flatten(1), weight.reshape(self.embed_dim, -1), bias)
```

Numerically identical, not an approximation: max abs difference 5.2e-6 against
the conv at batch 64 where MIOpen still works, which is float32 reduction-order
noise. It also skips MIOpen entirely rather than working around it by chunking.

This is **not H3-specific** — it is a general Qwen3-VL image-conditioning bug on
this GPU. Any Qwen3-VL vision input above ~256 patches (roughly, any image
larger than 256x256) hits it. Worth reporting upstream, separately from the
weight corruption above.

## Open threads

1. **Attention backend.** ComfyUI logs "Using pytorch attention". Whether a
   fused kernel is dispatching is still unknown, and it remains the most
   promising untried speedup — worth testing ComfyUI's alternate cross-attention
   flags and comparing the *constant*.

   **The measured exponent says nothing about this.** FlashAttention and friends
   are O(N²) in time too — they avoid materializing the N×N score matrix, which
   is a memory win, but every score still gets computed. Quadratic time is what
   *correct* attention looks like at any level of fusion; fusion moves the
   constant. An earlier version of this file inferred "quadratic, therefore
   unfused", which does not follow.
2. ~~**Unpruned weights.**~~ **Tested — no measurable difference.** Unpruned int8
   (34GB, `Comfy-Org/MiniMax-H3`) landed within noise of the pruned 21GB file at
   1.5x the per-step cost. Comfy-Org's copies are also *clean*: 1036 tensors,
   declared end exactly equal to file size, no repair pass needed — confirming
   the corruption below is Abiray's upload tooling, not upstream. One seed only,
   so a small effect is not excluded; 3–4 seeds per arm would settle it. The
   **int8 text encoder** (27GB vs int4) and **pruned bf16** (38GB, unquantized,
   vs pruned int8) were both tested too, and both are indistinguishable. Every
   component is now checked against higher precision along both axes — pruning
   and quantization — and none costs measurable quality. bf16 is also ~17%
   *faster* than int8 at 448x704, so int8 buys footprint rather than speed. All
   A/Bs are single-seed, which rules out large effects rather than small ones.
3. ~~**Finish a 1344x768 run.**~~ **Done — 56 min 47 s**, and 768x1344 portrait
   in 58 min 21 s. It did not confirm the quadratic model, it broke it: the
   top-end exponent is ~2.3, not 2.0, and there is no single exponent at all.
4. **Pin down the exponent properly.** A fourth point killed the single-exponent
   fit outright — pairwise exponents are now 1.24, 2.85, 2.33, non-monotonic.
   Two fixes are needed before more points help. First, **control for server
   age**: the same canvas measured 21.0 and 27.8 s/it depending on it, a spread
   wider than the effect being measured, so every run needs a fresh restart to
   be comparable. Second, `probe_comfy.py` still records wall time only —
   parsing the tqdm s/it out of the ComfyUI log into `timings.jsonl` would make
   every run contribute a comparable number for free. A point above 125k tokens
   needs longer clips, since 1344x768 is the canvas ceiling: 448x704 at 362
   frames is ~111k tokens and ~43 min, and 512x896 at 362 frames is ~161k but
   untested against the 89% VRAM ceiling the 125k runs already reach.
5. **Where does the super-quadratic term come from?** The cache-spill theory is
   untested. `rocperf`/`rocprof` counters on a small vs large canvas would show
   whether L2 hit rate collapses as the sequence grows, which would confirm it
   and suggest whether a tiled attention implementation could recover the loss.

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
