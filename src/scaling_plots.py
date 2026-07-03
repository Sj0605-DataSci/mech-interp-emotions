"""Cross-family scaling analysis: aggregate every model's geometry/logit-lens
metrics and plot them vs parameter count, coloured by family. This is the novel
contribution beyond the single-model paper — does emotion-concept structure
emerge, and sharpen, with scale, and does it differ across model families?

Reads out/<tag>/geometry.json (+ logitlens.json if present).
Writes out/scaling/*.png and out/scaling/summary.csv

Usage:
    python src/scaling_plots.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import OUT_ROOT, list_models

SCALING_DIR = OUT_ROOT / "scaling"

FAMILY_COLOR = {"pythia": "#1f77b4", "cerebras": "#d62728", "qwen": "#2ca02c"}


def parse_tag(tag: str):
    """(family, params_millions, nice_label) from a model dir tag."""
    t = tag.lower()
    m = re.search(r"(\d+\.?\d*)\s*([mb])", t)
    if not m:
        return None
    num, unit = float(m.group(1)), m.group(2)
    params_m = num * (1000 if unit == "b" else 1)
    if "pythia" in t:
        fam = "pythia"
    elif "cerebras" in t:
        fam = "cerebras"
    elif "qwen" in t:
        fam = "qwen"
    else:
        fam = "other"
    label = tag.split("__")[-1]
    return fam, params_m, label


def collect():
    rows = []
    for tag in list_models():
        gpath = OUT_ROOT / tag / "geometry.json"
        if not gpath.exists():
            continue
        g = json.loads(gpath.read_text())
        p = g["at_probe_layer"]
        cx = p.get("circumplex", {})
        parsed = parse_tag(tag)
        if not parsed:
            continue
        fam, params_m, label = parsed
        ll = OUT_ROOT / tag / "logitlens.json"
        ll_rate = json.loads(ll.read_text())["_self_hit_rate"] if ll.exists() else None
        rows.append({
            "tag": tag, "family": fam, "params_m": params_m, "label": label,
            "syn_ant_sep": p["syn_ant_sep"], "cluster_sep": p["cluster_sep"],
            "valence_r": abs(cx.get("valence", {}).get("r", 0.0)),
            "arousal_r": abs(cx.get("arousal", {}).get("r", 0.0)),
            "logitlens_rate": ll_rate,
        })
    rows.sort(key=lambda r: (r["family"], r["params_m"]))
    return rows


def plot_metric(rows, key, ylabel, fname, title):
    plt.figure(figsize=(7, 5))
    for fam in ("pythia", "cerebras", "qwen"):
        pts = [(r["params_m"], r[key]) for r in rows
               if r["family"] == fam and r.get(key) is not None]
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        plt.plot(xs, ys, "o-", color=FAMILY_COLOR[fam], label=fam, markersize=6)
    plt.xscale("log")
    plt.xlabel("Parameters (millions, log scale)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    SCALING_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(SCALING_DIR / fname, dpi=130)
    plt.close()


def main():
    rows = collect()
    if not rows:
        print("No geometry.json found. Run: python src/analyze_geometry.py --all")
        return
    SCALING_DIR.mkdir(parents=True, exist_ok=True)

    # CSV
    import csv
    keys = ["tag", "family", "params_m", "label", "syn_ant_sep", "cluster_sep",
            "valence_r", "arousal_r", "logitlens_rate"]
    with open(SCALING_DIR / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    plot_metric(rows, "syn_ant_sep", "synonym−antonym cosine separation",
                "scaling_syn_ant.png", "Emotion geometry quality vs scale")
    plot_metric(rows, "cluster_sep", "within−cross cluster cosine",
                "scaling_cluster.png", "Emotion cluster separation vs scale")
    plot_metric(rows, "valence_r", "|r| PC vs human valence",
                "scaling_valence.png", "Valence axis recovery vs scale")
    plot_metric(rows, "arousal_r", "|r| PC vs human arousal",
                "scaling_arousal.png", "Arousal axis recovery vs scale")
    plot_metric(rows, "logitlens_rate", "logit-lens self-token hit rate",
                "scaling_logitlens.png", "Logit-lens readout vs scale")

    print(f"Wrote {len(rows)} models -> {SCALING_DIR}/")
    print(f"{'model':<28}{'fam':<10}{'P(M)':>8}{'syn-ant':>9}{'clust':>7}{'val_r':>7}{'aro_r':>7}")
    for r in rows:
        print(f"{r['label']:<28}{r['family']:<10}{r['params_m']:>8.0f}"
              f"{r['syn_ant_sep']:>9.3f}{r['cluster_sep']:>7.3f}"
              f"{r['valence_r']:>7.2f}{r['arousal_r']:>7.2f}")


if __name__ == "__main__":
    main()
