"""Compute emotion vectors from per-story mean activations.

Recipe (faithful to the paper / reference repo, applied per layer):
  1. emotion mean  = mean over that emotion's stories
  2. emotion vector = emotion mean - global mean (mean across emotion means)
  3. confound removal: project out the top PCs of NEUTRAL activations that
     explain >= VARIANCE_THRESHOLD of neutral variance
  4. unit-normalise

Reads:  out/<tag>/story_means.pt, neutral_means.pt, meta.json
Writes: out/<tag>/emotion_vectors.pt
        {"vectors": {layer: {emotion: (d,)}}, "global_means": {layer:(d,)},
         "pcs": {layer:(k,d)}, "emotions":[...], "meta":{...}}

Usage:
    python src/compute_vectors.py --tag EleutherAI__pythia-70m-deduped
    python src/compute_vectors.py --repo EleutherAI/pythia-70m-deduped
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT / "out"
VARIANCE_THRESHOLD = 0.5


def top_pcs_for_variance(X: torch.Tensor, threshold: float) -> torch.Tensor:
    """Top PCs (orthonormal rows, shape (k, D)) of X (rows=samples) covering at
    least `threshold` of total variance. SVD for numerical stability."""
    Xc = X - X.mean(dim=0, keepdim=True)
    _, S, Vt = torch.linalg.svd(Xc, full_matrices=False)
    var = S.pow(2)
    cum = var.cumsum(0) / var.sum()
    k = int((cum < threshold).sum().item()) + 1
    return Vt[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", help="model tag (dir under out/)")
    ap.add_argument("--repo", help="alternative to --tag; will be slugified")
    ap.add_argument("--variance", type=float, default=VARIANCE_THRESHOLD)
    ap.add_argument("--n-stories", type=int, default=300,
                    help="stories/emotion used to form each class mean. Fixed across "
                         "ALL models for a fair cross-family comparison. 300 is "
                         "cosine-0.9998 identical to the full 1200 (verified), so "
                         "lossless. Set 0 to use all available.")
    args = ap.parse_args()

    tag = args.tag or args.repo.replace("/", "__")
    out_dir = OUT_ROOT / tag
    meta = json.loads((out_dir / "meta.json").read_text())
    L, d = meta["n_layers"], meta["d_model"]

    story_means = torch.load(out_dir / "story_means.pt", weights_only=True)   # {emo:(n,L,d)}
    neutral = torch.load(out_dir / "neutral_means.pt", weights_only=True)     # (n,L,d)
    emotions = sorted(story_means)
    N = args.n_stories or None
    avail = min(v.shape[0] for v in story_means.values())
    if N and avail < N:
        print(f"  WARNING: only {avail} stories/emotion available (< {N}); using all {avail}")
        N = avail
    print(f"{tag}: {len(emotions)} emotions, {L} layers, d={d}, "
          f"stories/emotion={N or avail} (of {avail} extracted)")

    vectors, global_means, pcs_per_layer = {}, {}, {}
    for layer in range(L):
        # emotion means at this layer over the first N stories: (E, d)
        emo_means = {e: story_means[e][:N, layer, :].mean(dim=0) for e in emotions}
        gmean = torch.stack(list(emo_means.values())).mean(dim=0)
        global_means[layer] = gmean

        pcs = top_pcs_for_variance(neutral[:, layer, :].float(), args.variance)  # (k,d)
        pcs_per_layer[layer] = pcs

        vlayer = {}
        for e, m in emo_means.items():
            v = m - gmean
            v = v - pcs.T @ (pcs @ v)          # project out neutral PC subspace
            vlayer[e] = v / v.norm()
        vectors[layer] = vlayer
        if layer % max(1, L // 8) == 0:
            print(f"  layer {layer:>2}: stripped {pcs.shape[0]} neutral PCs")

    out = {"vectors": vectors, "global_means": global_means, "pcs": pcs_per_layer,
           "emotions": emotions, "meta": meta, "variance_threshold": args.variance,
           "n_stories": N or avail}
    torch.save(out, out_dir / "emotion_vectors.pt")
    size = (out_dir / "emotion_vectors.pt").stat().st_size / 1e6
    print(f"Saved emotion_vectors.pt ({size:.1f} MB)")


if __name__ == "__main__":
    main()
