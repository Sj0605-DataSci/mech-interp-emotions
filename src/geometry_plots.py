"""Generate all paper Part-2 geometry figures for each model.

Figures produced per model (in out/<tag>/geometry_plots/):
  fig5_cosine_heatmap.png  — pairwise cosine similarity with hierarchical clustering
  fig6_umap_clusters.png   — UMAP of k=10 k-means clusters
  fig7_pca_bars.png        — per-emotion PC1/PC2 projection bar chart
  fig8_pc_scatter.png      — PC1 vs human valence, PC2 vs human arousal scatter
  fig9_rsa_heatmap.png     — cross-layer RSA matrix (structure stability)

Usage:
    python src/geometry_plots.py --tag EleutherAI__pythia-2.8b-deduped
    python src/geometry_plots.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))

from analysis_common import (EMOTION_CLUSTERS, OUT_ROOT,
                             cosine_matrix, list_models, load_vectors,
                             probe_layer, vector_matrix)
from human_vad import HUMAN_VAD

try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.cluster import KMeans
    HAS_SK = True
except ImportError:
    HAS_SK = False

# ── colour palette ──────────────────────────────────────────────────────────────
CLUSTER_COLORS = [
    "#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00",
    "#a65628","#f781bf","#999999","#e6ab02","#66c2a5",
]


def _pca(M: np.ndarray):
    """Returns (projections, variance_ratios, components)."""
    X = M - M.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    proj = U * S          # (n, k)
    var_ratio = S**2 / (S**2).sum()
    return proj, var_ratio, Vt


# ── Fig 5: pairwise cosine heatmap ─────────────────────────────────────────────
def fig5_cosine_heatmap(tag: str, out_dir: Path):
    data = load_vectors(tag)
    layer = probe_layer(data)
    emos, M = vector_matrix(data, layer)
    C = cosine_matrix(M).numpy()
    n = len(emos)

    # hierarchical clustering on 1-cosine distances (clamp negatives → small positive)
    dist = np.clip(1 - C, 0, 2)
    np.fill_diagonal(dist, 0)
    linkage_mat = linkage(squareform(dist), method="average")
    order = dendrogram(linkage_mat, no_plot=True)["leaves"]

    C_ord = C[np.ix_(order, order)]
    labels_ord = [emos[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(C_ord, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    # show only every 5th tick for readability
    step = max(1, n // 30)
    ticks = list(range(0, n, step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([labels_ord[i] for i in ticks], rotation=90, fontsize=6)
    ax.set_yticklabels([labels_ord[i] for i in ticks], fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.04, label="cosine similarity")
    ax.set_title(f"Fig 5 — Pairwise cosine similarity (hierarchically clustered)\n{tag.split('__')[-1]}",
                 fontsize=10)
    fig.tight_layout()
    path = out_dir / "fig5_cosine_heatmap.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Fig 6: UMAP of k=10 clusters ───────────────────────────────────────────────
def fig6_umap_clusters(tag: str, out_dir: Path):
    data = load_vectors(tag)
    layer = probe_layer(data)
    emos, M = vector_matrix(data, layer)
    X = M.numpy()
    path = out_dir / "fig6_umap_clusters.png"

    if not HAS_UMAP or not HAS_SK:
        # fallback: PCA scatter instead
        proj, _, _ = _pca(X)
        fig, ax = plt.subplots(figsize=(9, 7))
        km = KMeans(n_clusters=10, random_state=42, n_init=10).fit(X) if HAS_SK else None
        labels = km.labels_ if km else np.zeros(len(emos), dtype=int)
        for k in range(10):
            mask = labels == k
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=CLUSTER_COLORS[k % 10], alpha=0.75, s=40, label=f"C{k}")
        # annotate a subset
        for i, e in enumerate(emos):
            if i % 8 == 0:
                ax.annotate(e, (proj[i, 0], proj[i, 1]), fontsize=6, alpha=0.7)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title(f"Fig 6 — Emotion clusters (PCA, k=10 k-means)\n{tag.split('__')[-1]}", fontsize=9)
        ax.legend(fontsize=7, markerscale=0.8, ncol=2)
        fig.tight_layout()
        fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
        return path

    if HAS_SK:
        km = KMeans(n_clusters=10, random_state=42, n_init=10).fit(X)
        labels = km.labels_
    else:
        labels = np.zeros(len(emos), dtype=int)

    reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(emos)-1))
    emb = reducer.fit_transform(X)

    # cluster names (paper uses Claude to name — we use the EMOTION_CLUSTERS keys)
    cluster_names = list(EMOTION_CLUSTERS.keys())

    fig, ax = plt.subplots(figsize=(10, 8))
    for k in range(10):
        mask = labels == k
        name = cluster_names[k] if k < len(cluster_names) else f"C{k}"
        ax.scatter(emb[mask, 0], emb[mask, 1],
                   c=CLUSTER_COLORS[k % 10], alpha=0.8, s=45, label=name)
    # label every 6th point
    for i, e in enumerate(emos):
        if i % 6 == 0:
            ax.annotate(e, (emb[i, 0], emb[i, 1]), fontsize=6, alpha=0.75,
                        xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title(f"Fig 6 — UMAP of emotion vectors (k=10 k-means)\n{tag.split('__')[-1]}", fontsize=9)
    ax.legend(fontsize=7, markerscale=0.8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Fig 7: PCA bar chart ───────────────────────────────────────────────────────
def fig7_pca_bars(tag: str, out_dir: Path):
    data = load_vectors(tag)
    layer = probe_layer(data)
    emos, M = vector_matrix(data, layer)
    proj, var_ratio, _ = _pca(M.numpy())

    # sort by PC1 projection for the bar chart
    order_pc1 = np.argsort(proj[:, 0])
    order_pc2 = np.argsort(proj[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(16, max(6, len(emos) * 0.14)))
    for ax, order, pc_idx, label in [
        (axes[0], order_pc1, 0, f"PC1 ({var_ratio[0]*100:.0f}% var)"),
        (axes[1], order_pc2, 1, f"PC2 ({var_ratio[1]*100:.0f}% var)"),
    ]:
        vals = proj[order, pc_idx]
        names = [emos[i] for i in order]
        colors = ["#d73027" if v < 0 else "#4575b4" for v in vals]
        ax.barh(range(len(names)), vals, color=colors, height=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=5.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("projection value")
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Fig 7 — PCA projections of emotion vectors\n{tag.split('__')[-1]}", fontsize=9)
    fig.tight_layout()
    path = out_dir / "fig7_pca_bars.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Fig 8: PC scatter vs human VAD ─────────────────────────────────────────────
def fig8_pc_scatter(tag: str, out_dir: Path):
    data = load_vectors(tag)
    layer = probe_layer(data)
    emos, M = vector_matrix(data, layer)

    keep = [(i, e) for i, e in enumerate(emos) if e in HUMAN_VAD]
    if len(keep) < 8:
        return None
    idxs = [i for i, _ in keep]
    names = [e for _, e in keep]
    Msub = M[idxs].numpy()
    proj, var_ratio, _ = _pca(Msub)

    val_h = np.array([HUMAN_VAD[e][0] for e in names])
    aro_h = np.array([HUMAN_VAD[e][1] for e in names])

    # find best PC for each axis
    def best_pc(target):
        best_r, best_k = 0.0, 0
        for k in range(min(4, proj.shape[1])):
            p = proj[:, k]
            if p.std() < 1e-9: continue
            r = float(np.corrcoef(p, target)[0, 1])
            if abs(r) > abs(best_r):
                best_r, best_k = r, k
        return best_k, best_r

    val_pc, val_r = best_pc(val_h)
    aro_pc, aro_r = best_pc(aro_h)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, pc_idx, h_vals, h_label, r in [
        (axes[0], val_pc, val_h, "Human valence", val_r),
        (axes[1], aro_pc, aro_h, "Human arousal", aro_r),
    ]:
        x = proj[:, pc_idx]
        ax.scatter(x, h_vals, alpha=0.7, s=30, c="#3b7ec8")
        for i, name in enumerate(names):
            if i % 4 == 0:
                ax.annotate(name, (x[i], h_vals[i]), fontsize=6, alpha=0.7,
                            xytext=(2, 2), textcoords="offset points")
        # fit line
        m, b = np.polyfit(x, h_vals, 1)
        xl = np.linspace(x.min(), x.max(), 100)
        ax.plot(xl, m*xl + b, "r--", lw=1)
        ax.set_xlabel(f"PC{pc_idx+1} projection ({var_ratio[pc_idx]*100:.0f}% var)")
        ax.set_ylabel(h_label)
        ax.set_title(f"r = {r:+.2f}", fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)

    fig.suptitle(f"Fig 8 — Emotion PCs vs human norms\n{tag.split('__')[-1]}", fontsize=9)
    fig.tight_layout()
    path = out_dir / "fig8_pc_scatter.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Fig 9: cross-layer RSA heatmap ─────────────────────────────────────────────
def fig9_rsa_heatmap(tag: str, out_dir: Path):
    data = load_vectors(tag)
    L = data["meta"]["n_layers"]
    layer = probe_layer(data)

    # sample up to 20 evenly-spaced layers
    stride = max(1, L // 20)
    layers = list(range(0, L, stride))

    flats = []
    for lyr in layers:
        _, M = vector_matrix(data, lyr)
        C = cosine_matrix(M).numpy()
        iu = np.triu_indices_from(C, k=1)
        flats.append(C[iu])
    flats = np.stack(flats)
    rsa = np.corrcoef(flats)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(rsa, vmin=0, vmax=1, cmap="viridis", aspect="auto",
                   extent=[layers[0]-0.5, layers[-1]+0.5, layers[-1]+0.5, layers[0]-0.5])
    plt.colorbar(im, ax=ax, fraction=0.04, label="Pearson r")
    ax.set_xlabel("Layer"); ax.set_ylabel("Layer")
    # mark probe layer
    if layer in layers:
        p_idx = layers.index(layer)
        ax.axhline(layers[p_idx] - 0.5, color="white", lw=1, ls="--", alpha=0.8)
        ax.axvline(layers[p_idx] - 0.5, color="white", lw=1, ls="--", alpha=0.8)
    off_diag = rsa[~np.eye(len(layers), dtype=bool)].mean()
    ax.set_title(f"Fig 9 — Cross-layer RSA (off-diag mean={off_diag:.3f})\n{tag.split('__')[-1]}",
                 fontsize=9)
    fig.tight_layout()
    path = out_dir / "fig9_rsa_heatmap.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ── driver ──────────────────────────────────────────────────────────────────────
def run_tag(tag: str):
    out_dir = OUT_ROOT / tag / "geometry_plots"
    out_dir.mkdir(exist_ok=True)
    label = tag.split("__")[-1]
    print(f"=== {label}")

    p5 = fig5_cosine_heatmap(tag, out_dir)
    print(f"  fig5 done")
    p6 = fig6_umap_clusters(tag, out_dir)
    print(f"  fig6 done")
    p7 = fig7_pca_bars(tag, out_dir)
    print(f"  fig7 done")
    p8 = fig8_pc_scatter(tag, out_dir)
    print(f"  fig8 done")
    p9 = fig9_rsa_heatmap(tag, out_dir)
    print(f"  fig9 done  →  {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    tags = list_models() if args.all else ([args.tag] if args.tag else [])
    if not tags:
        print("Use --tag <tag> or --all"); return
    for tag in tags:
        run_tag(tag)


if __name__ == "__main__":
    main()
