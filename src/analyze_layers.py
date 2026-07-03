"""Paper Part-2 layer analyses: how emotion-concept representation evolves with
DEPTH, and how stable its structure is across layers.

(1) Layer dynamics: syn-ant separation / circumplex |r| as a function of relative
    depth. The paper finds emotion concepts are weak in early layers (surface
    "sensory" features), strengthen through the middle, and the "operative"
    emotion for predicting upcoming tokens lives in mid-late layers.

(2) Cross-layer RSA (representational similarity analysis): for each pair of
    layers, correlate their 171x171 emotion cosine-similarity matrices. High
    off-diagonal => the *structure* of emotion space is stable across depth
    (paper Fig 9: stable from early-mid to late layers).

Reads out/<tag>/geometry.json (per-layer syn/ant/circumplex already computed) +
emotion_vectors.pt (for the RSA matrices). Writes out/<tag>/layers.json and
out/layers/<tag>_*.png.

Usage:
    python src/analyze_layers.py --tag EleutherAI__pythia-2.8b-deduped
    python src/analyze_layers.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import (OUT_ROOT, cosine_matrix, list_models, load_vectors,
                             vector_matrix)

LAYERS_DIR = OUT_ROOT / "layers"


def layer_dynamics(tag: str):
    """Per-layer syn-ant separation + circumplex |r|, from geometry.json."""
    g = json.loads((OUT_ROOT / tag / "geometry.json").read_text())
    pl = g["per_layer"]
    L = g["n_layers"]
    rows = []
    for layer in range(L):
        d = pl[str(layer)] if str(layer) in pl else pl[layer]
        cx = d.get("circumplex", {})
        rows.append({
            "layer": layer, "rel_depth": layer / (L - 1) if L > 1 else 0,
            "syn_ant_sep": d["syn_ant_sep"],
            "valence_r": abs(cx.get("valence", {}).get("r", 0.0)),
            "arousal_r": abs(cx.get("arousal", {}).get("r", 0.0)),
        })
    return rows


def cross_layer_rsa(data: dict, n_layers: int, stride: int = 1) -> np.ndarray:
    """(Ls x Ls) matrix: correlation between layers' emotion cosine-sim matrices."""
    layers = list(range(0, n_layers, stride))
    # upper-triangular (excl diagonal) of each layer's cosine matrix, flattened
    flats = []
    for layer in layers:
        _, M = vector_matrix(data, layer)
        C = cosine_matrix(M).numpy()
        iu = np.triu_indices_from(C, k=1)
        flats.append(C[iu])
    flats = np.stack(flats)            # (Ls, n_pairs)
    rsa = np.corrcoef(flats)           # (Ls, Ls)
    return layers, rsa


def analyze_tag(tag: str, plot: bool = True):
    data = load_vectors(tag)
    L = data["meta"]["n_layers"]
    dyn = layer_dynamics(tag)
    stride = max(1, L // 20)           # cap RSA at ~20 layers for big models
    rsa_layers, rsa = cross_layer_rsa(data, L, stride)
    # off-diagonal mean = how stable structure is across depth
    off = rsa[~np.eye(len(rsa), dtype=bool)]
    out = {"tag": tag, "n_layers": L, "layer_dynamics": dyn,
           "rsa_layers": rsa_layers, "rsa_offdiag_mean": float(off.mean())}
    (OUT_ROOT / tag / "layers.json").write_text(json.dumps(out, indent=2))

    if plot:
        LAYERS_DIR.mkdir(parents=True, exist_ok=True)
        # depth curve
        rd = [r["rel_depth"] for r in dyn]
        plt.figure(figsize=(7, 4.5))
        plt.plot(rd, [r["syn_ant_sep"] for r in dyn], "o-", label="syn-ant sep")
        plt.plot(rd, [r["arousal_r"] for r in dyn], "s-", label="arousal |r|")
        plt.plot(rd, [r["valence_r"] for r in dyn], "^-", label="valence |r|")
        plt.xlabel("relative depth (0=first layer, 1=last)")
        plt.ylabel("metric"); plt.title(f"Emotion representation vs depth — {tag.split('__')[-1]}")
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(LAYERS_DIR / f"{tag}_depth.png", dpi=120); plt.close()
        # RSA heatmap
        plt.figure(figsize=(5.5, 4.5))
        plt.imshow(rsa, vmin=0, vmax=1, cmap="viridis",
                   extent=[rsa_layers[0], rsa_layers[-1], rsa_layers[-1], rsa_layers[0]])
        plt.colorbar(label="cosine-matrix correlation")
        plt.xlabel("layer"); plt.ylabel("layer")
        plt.title(f"Cross-layer RSA — {tag.split('__')[-1]}")
        plt.tight_layout(); plt.savefig(LAYERS_DIR / f"{tag}_rsa.png", dpi=120); plt.close()

    # peak-depth summary: where is syn-ant separation maximal?
    peak = max(dyn, key=lambda r: r["syn_ant_sep"])
    print(f"=== {tag}  (L={L})")
    print(f"  peak syn-ant sep at rel_depth {peak['rel_depth']:.2f} (layer {peak['layer']}): {peak['syn_ant_sep']:.3f}")
    print(f"  cross-layer RSA off-diag mean: {out['rsa_offdiag_mean']:.3f}  (1=identical structure all layers)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    tags = list_models() if args.all else [args.tag]
    if not tags or tags == [None]:
        print("Use --tag <tag> or --all. Available:", list_models()); return
    for tag in tags:
        analyze_tag(tag)


if __name__ == "__main__":
    main()
