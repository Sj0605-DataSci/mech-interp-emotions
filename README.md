# Mech Interp: Emotion Concepts (OSS scaling replication)

Replicating Anthropic's [*Emotion Concepts and their Function in a Large Language
Model*](https://transformer-circuits.pub/2026/emotions/index.html) (Sofroniew et
al., 2026) on open-source **base** models, as a **scaling study** across three
families.

## Model families (base / pretrained only)

The paper argues emotion concepts are inherited from *pretraining*; studying base
models tests that directly and keeps families comparable.

| Family | model_type | Sizes |
|---|---|---|
| Pythia (deduped) | `gpt_neox` | 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b |
| Cerebras-GPT | `gpt2` | 111M, 256M, 590M, 1.3B, 2.7B, 6.7B, 13B |
| Qwen3.5 dense Base | `qwen3_5` | 0.8B, 2B, 4B, 9B |

## Dataset

[`ryancodrai/emotion-probes`](https://huggingface.co/datasets/ryancodrai/emotion-probes)
— a faithful regeneration of the paper's stimuli: **171 emotions × 100 topics ×
12 = 205,200 stories** (emotion conveyed, never named) + 1,200 neutral stories
(for PCA confound removal) + deflection dialogues. Model-agnostic text, so the
same dataset feeds every model.

## Method (per model)

1. **extract** — run the 205k stories through the model, capture residual-stream
   activations at every decoder layer, average tokens from offset 50 onward.
   One compact `(n_stories, L, d)` tensor per emotion (NOT per-token => avoids
   the reference repo's 83 GB/model blowup).
2. **compute_vectors** — per layer: `emotion_mean − global_mean`, project out the
   top neutral PCs (≥50% variance), unit-normalise. => `emotion_vectors.pt`.
3. **analyse** (next) — geometry (cosine clustering, PCA→valence/arousal vs
   NRC-VAD), logit-lens, probing AUC vs GoEmotions, steering / causal effects,
   then plot every metric **vs model size** across the three families.

## Layout

```
src/
  layer_paths.py       per-family module-path registry (gpt_neox / gpt2 / qwen3_5)
  model_utils.py       model loader + residual-stream hook harness
  extract.py           stories -> per-story mean activations (memory-light)
  compute_vectors.py   per-story means -> emotion vectors (diff-of-means + neutral PCA)
  analysis_common.py   shared: clusters (k=10), projection, syn/antonym pairs
  analyze_geometry.py  cosine/cluster separation + PCA circumplex vs human VAD
  analyze_logitlens.py emotion vector -> unembed top tokens (paper Table 1)
  steer.py             activation steering (causal test) demo
  scaling_plots.py     cross-family metrics vs param count -> out/scaling/*.png
scripts/
  download_models.sh   download base models + dataset (uses venv `hf` + HF_TOKEN)
  run_all.sh           extract -> compute_vectors for every model (skips done)
  analyze_all.sh       geometry + logit-lens + scaling plots for every model
data/emotions_171.txt  the 171 emotion words
data/human_vad.py      human valence/arousal norms (circumplex reference)
reference_repo/         RyanCodrai's Gemma replication (reference only; Flask viz)
out/<tag>/             per-model: *_means.pt, emotion_vectors.pt, geometry.json, logitlens.json
out/scaling/           scaling_*.png + summary.csv (the headline result)
paper.txt              extracted prose of the paper
```

## Performance on GB10 / DGX Spark (sm_121)

Qwen3.5 extraction is the slow part (Gated DeltaNet linear attention has no fast
kernel on sm_121). What we found, after exhaustive testing:

**Helps (all applied):**
- **GPU clock lock** `sudo nvidia-smi -lgc 2220` — +~24% over the conservative
  1794 floor. The box thermal-shuts-down at 3003 MHz/96°C, so 2220 is the safe
  middle (sustained ~76°C). Find it with `scripts/find_safe_clock.sh`.
- **`cudnn.benchmark=True` + TF32** — ~7%, free, lossless. Baked into `model_utils.py`.
- **`fla` (flash-linear-attention)** installed via uv — makes Qwen run at all.

**Does NOT help (tested + rejected):**
- torch.compile (recompiles on varying seq len → eager fallback; fixed-shape pad
  costs more than it saves), bigger batches (bs64 optimal), sdpa/TF32 on attn
  (already on sdpa), HF `kernels` pkg (breaks transformers import), GB10 sm_121
  kernels (RMSNorm/GELU only), FlashQLA-Blackwell (only helps at 32k seq; ours ~175).
- Root cause (measured): our bf16 GEMM actually hits **83-92 TFLOPS** here (the
  "11 TFLOPS" blog numbers are under-clocked/fp32), so matmul is NOT the wall.
  The drag is bandwidth/latency-bound work — the GDN linear-attention recurrence
  (many small sequential ops that don't fill tensor cores) + per-token norms,
  limited by GB10's 273 GB/s. No PyTorch/kernel swap fixes a bandwidth limit;
  only less work (fewer stories/tokens) or lower precision would.

**PyTorch is already optimal for sm_121:** torch 2.12.1+cu130, native sm_120
cubins (binary-compatible w/ sm_121), CUDA 13 + cuDNN 9.2, no PTX-JIT, immune to
the NVRTC footgun (uses system NVRTC). Nothing to upgrade.

## Analysis layer

The reference repo only ships a Flask per-token heatmap viewer. We add the
paper's quantitative analyses + the cross-family scaling study it lacks:

- **Geometry** (`analyze_geometry.py`): per layer, synonym vs antonym cosine
  separation, k=10 cluster separation, and PCA whose top components are
  correlated against human valence/arousal (the affective circumplex).
- **Logit-lens** (`analyze_logitlens.py`): each emotion vector through final-norm
  + unembed; top up/down tokens + a self-token hit rate.
- **Steering** (`steer.py`): add alpha%·resid-norm of an emotion vector at every
  layer and compare continuations (causal check).
- **Scaling** (`scaling_plots.py`): every metric vs params, coloured by family.

## Resumable extraction

`extract.py` checkpoints each emotion to `out/<tag>/ckpt/<emotion>.pt` as it
finishes (atomic write). If killed mid-model, just re-run the same command — it
prints `resuming: N already checkpointed, M to go` and continues from the next
emotion (only the in-flight emotion, ~13s, is lost). Checkpoints are deleted
after a full successful run (`--keep-ckpt` to keep them); they're auto-discarded
if extraction params (N / token offset / max_len) change, so a resume never mixes
runs. Memory is bounded per batch (`cap.clear()` each iteration) — required on
GB10's shared 128GB pool, where the old all-on-GPU capture ballooned the 13B to
84GB and OOM-killed the desktop.

## Usage

```bash
# 1. download everything (base models + dataset)
HF_TOKEN=hf_xxx ./scripts/download_models.sh all

# 2. extract + compute vectors for all models (smallest first)
./scripts/run_all.sh
# quick smoke run:  LIMIT=16 ./scripts/run_all.sh

# single model:
python src/extract.py --repo EleutherAI/pythia-70m-deduped
python src/compute_vectors.py --repo EleutherAI/pythia-70m-deduped

# 3. analysis + scaling plots (after vectors exist)
./scripts/analyze_all.sh
# or piecewise:
python src/analyze_geometry.py --all
python src/analyze_logitlens.py --repo EleutherAI/pythia-70m-deduped
python src/scaling_plots.py
python src/steer.py --repo EleutherAI/pythia-70m-deduped --emotion desperate \
    --prompt "I opened the email and" --alpha 8
```

Outputs land in `out/<repo_slug>/emotion_vectors.pt`:
`{"vectors": {layer: {emotion: (d,)}}, "global_means", "pcs", "emotions", "meta"}`.

## Status

- [x] Paper read, dataset + base models identified
- [x] Pipeline: `layer_paths` / `model_utils` / `extract` / `compute_vectors`
- [x] Validated end-to-end on pythia-70m (synonyms cluster +, opposites −)
- [x] Analysis layer built + validated (geometry, logit-lens, steering, scaling plots)
- [ ] Full run across all 19 models (`run_all.sh`), then `analyze_all.sh`
- [ ] (later) GoEmotions probing AUC, deflection probes, alignment-behavior steering
