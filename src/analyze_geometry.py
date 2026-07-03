"""Per-model geometry analysis (paper Part 2): does the emotion-vector space
recover human emotional structure?

For each layer we compute:
  - synonym mean cosine, antonym mean cosine, and their separation (a clean,
    label-free "is the geometry meaningful?" scalar)
  - PCA of the 171 emotion vectors; correlate PC1/PC2 with human valence/arousal
    (the affective-circumplex test). We report the BEST |r| over the top few PCs
    for each axis, since which PC carries valence vs arousal can swap by layer.
  - cluster separation: mean within-cluster cosine minus mean cross-cluster cosine
    using the paper's k=10 groups.

Writes out/<tag>/geometry.json and prints a summary at the probe layer.

Usage:
    python src/analyze_geometry.py --tag EleutherAI__pythia-70m-deduped
    python src/analyze_geometry.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from analysis_common import (ANTONYM_PAIRS, EMOTION_CLUSTERS, SYNONYM_PAIRS,
                             cosine_matrix, list_models, load_vectors,
                             probe_layer, vector_matrix)
from human_vad import HUMAN_VAD


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def pair_cosines(emos, C, pairs):
    idx = {e: i for i, e in enumerate(emos)}
    vals = [C[idx[a], idx[b]].item() for a, b in pairs if a in idx and b in idx]
    return float(np.mean(vals)) if vals else float("nan")


def cluster_separation(emos, C) -> float:
    idx = {e: i for i, e in enumerate(emos)}
    within, cross = [], []
    members = {g: [idx[e] for e in es if e in idx] for g, es in EMOTION_CLUSTERS.items()}
    groups = list(members)
    for gi, g in enumerate(groups):
        mi = members[g]
        for a in range(len(mi)):
            for b in range(a + 1, len(mi)):
                within.append(C[mi[a], mi[b]].item())
        for h in groups[gi + 1:]:
            for x in mi:
                for y in members[h]:
                    cross.append(C[x, y].item())
    if not within or not cross:
        return float("nan")
    return float(np.mean(within) - np.mean(cross))


def circumplex_corr(emos, M):
    """PCA the emotion vectors; return best |r| of any top-4 PC with human
    valence and with arousal, plus which PC and signed r."""
    # restrict to emotions we have human norms for
    keep = [(i, e) for i, e in enumerate(emos) if e in HUMAN_VAD]
    if len(keep) < 10:
        return {}
    idxs = [i for i, _ in keep]
    names = [e for _, e in keep]
    X = M[idxs].numpy()
    X = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pcs = U * S  # projections (n, k)
    val = np.array([HUMAN_VAD[e][0] for e in names])
    aro = np.array([HUMAN_VAD[e][1] for e in names])
    out = {}
    for axis_name, target in (("valence", val), ("arousal", aro)):
        best = (0.0, -1)
        for k in range(min(4, pcs.shape[1])):
            r = _pearson(pcs[:, k], target)
            if abs(r) > abs(best[0]):
                best = (r, k)
        out[axis_name] = {"r": round(best[0], 4), "pc": best[1]}
    var = (S ** 2)
    out["pc_var_ratio"] = [round(float(v), 4) for v in (var / var.sum())[:4]]
    out["n_emotions_matched"] = len(names)
    return out


def analyze_tag(tag: str) -> dict:
    data = load_vectors(tag)
    L = data["meta"]["n_layers"]
    pl = probe_layer(data)
    per_layer = {}
    for layer in range(L):
        emos, M = vector_matrix(data, layer)
        C = cosine_matrix(M)
        syn = pair_cosines(emos, C, SYNONYM_PAIRS)
        ant = pair_cosines(emos, C, ANTONYM_PAIRS)
        per_layer[layer] = {
            "syn_cos": round(syn, 4),
            "ant_cos": round(ant, 4),
            "syn_ant_sep": round(syn - ant, 4),
            "cluster_sep": round(cluster_separation(emos, C), 4),
            "circumplex": circumplex_corr(emos, M),
        }
    result = {"tag": tag, "n_layers": L, "d_model": data["meta"]["d_model"],
              "probe_layer": pl, "per_layer": per_layer,
              "at_probe_layer": per_layer[pl]}
    (analysis_common_out(tag) / "geometry.json").write_text(json.dumps(result, indent=2))
    return result


def analysis_common_out(tag: str) -> Path:
    from analysis_common import OUT_ROOT
    return OUT_ROOT / tag


def print_summary(r: dict):
    p = r["at_probe_layer"]
    cx = p.get("circumplex", {})
    print(f"\n=== {r['tag']}  (L={r['n_layers']}, d={r['d_model']}, probe layer {r['probe_layer']})")
    print(f"  synonym cos        : {p['syn_cos']:+.3f}")
    print(f"  antonym cos        : {p['ant_cos']:+.3f}")
    print(f"  syn-ant separation : {p['syn_ant_sep']:+.3f}   (higher = cleaner geometry)")
    print(f"  cluster separation : {p['cluster_sep']:+.3f}")
    if cx:
        print(f"  PC~valence  |r|    : {cx['valence']['r']:+.3f} (PC{cx['valence']['pc']})")
        print(f"  PC~arousal  |r|    : {cx['arousal']['r']:+.3f} (PC{cx['arousal']['pc']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    tags = list_models() if args.all else [args.tag]
    if not tags or tags == [None]:
        print("No models. Use --tag <tag> or --all. Available:", list_models())
        return
    for tag in tags:
        print_summary(analyze_tag(tag))


if __name__ == "__main__":
    main()
