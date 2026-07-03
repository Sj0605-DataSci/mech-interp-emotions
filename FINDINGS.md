# Emotion Concepts Across Scale — Findings

Replication + scaling study of Anthropic's *Emotion Concepts and their Function in
a Large Language Model* (Sofroniew et al., 2026) on **19 open base models** across
**3 families** (Pythia, Cerebras-GPT, Qwen3.5), all probed at a fixed **N=300
stories/emotion**, vectors taken ~2/3 depth.

Plots: `out/scaling/*.png` (cross-family), `out/layers/*.png` (per-model depth+RSA).
Data: `out/scaling/summary.csv`, `out/<tag>/{geometry,layers}.json`.

## Part 1 — emotion vectors exist and are meaningful
Difference-of-means emotion vectors (neutral-PCA denoised) cleanly separate
synonyms from antonyms in every model ≥160M. Synonym cos ~0.85, antonym cos ~−0.35
at the largest scales.

## Part 2 — geometry mirrors human emotion (the paper's core claim, reproduced in OSS)

**Affective circumplex.** PCA of the 171 emotion vectors recovers human structure:
- **PC1 ≈ arousal**: |r| vs human norms rises 0.62 (pythia-70m) → **0.79** (large).
- **PC2 ≈ valence**: |r| ≈ 0.65, essentially flat across ALL sizes (learned earliest).
Clusters (k=10) group intuitively (joy/excitement, fear/anxiety, sadness/grief).

**Layer dynamics.** Emotion separation is weak in early layers and peaks **deep**
(rel-depth 0.65→1.0, deeper as models grow) — consistent with the paper's
"operative emotion for predicting upcoming tokens lives in mid-late layers."

**Cross-layer stability (RSA).** The 171×171 emotion-similarity structure is highly
consistent across depth (off-diagonal r = **0.95–0.98** every model), reproducing
the paper's Fig-9 "stable from early-mid to late layers."

## The scaling result (our novel contribution)

**Emotion-concept representation is an EARLY-SATURATING capability.** Every metric
rises steeply ~100M→1B, then **plateaus** — bigger models barely improve.

| metric | 100M | ~1B | 2–3B | 7–13B |
|---|---|---|---|---|
| syn−ant separation | 0.85–0.87 | 1.18–1.20 | 1.21–1.23 | **1.19–1.22 (flat)** |
| arousal \|r\| | 0.62–0.72 | 0.75 | 0.78 | **0.79 (flat)** |
| cluster separation | 0.32–0.35 | 0.41 | 0.43–0.44 | **0.44–0.45 (flat)** |

**Is the trend flat? Yes** — clean saturation by ~1–3B params. A ~1B model already
represents emotion concepts about as well as a 12–13B one. This contrasts with
capabilities (reasoning, knowledge) that keep scaling, and says emotion is
structurally simple: enough capacity to lay 171 concepts on a valence/arousal
manifold, which ~1B params supplies.

**Cross-family convergence.** All three families — different data (Pile / dedup-Pile
/ Qwen corpus) and architectures (GPT-NeoX / GPT-2 / Gated-DeltaNet) — converge to
the SAME plateau (~1.20 sep, ~0.79 arousal). Emotion-concept geometry is a
training-invariant, universal property of LM scale, strong evidence it is inherited
from generic pretraining (the paper's central thesis), now shown beyond Claude.

## Still TODO
- Logit-lens scaling (does vocab readout of emotions sharpen with scale? — small
  models showed weak readout despite good geometry).
- Steering / causal effects (the functional half: do emotion vectors *drive*
  behavior, and does that scale?).
