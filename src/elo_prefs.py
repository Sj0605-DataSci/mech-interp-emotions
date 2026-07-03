"""Paper Part 1 (end): Activity preference Elo experiment.

Replicates Sofroniew et al. Fig 4: Does emotion probe activation on an activity
description predict the model's PREFERENCE for that activity?

Method (adapted for base models — no chat template):
  1. Define N activities across valence categories (helpful, neutral, aversive).
  2. For each activity, compute emotion probe activations at the probe layer on
     the activity token span.
  3. Run all N*(N-1)/2 pairwise comparisons: which activity does the model
     prefer? Measured by comparing next-token log-prob for "A" vs "B" after
     a preference prompt.
  4. Compute Elo scores from pairwise outcomes.
  5. For each emotion probe, correlate probe activation with Elo score.
     High |r| = this emotion tracks model preferences.

Writes: out/<tag>/elo_prefs.json
        out/<tag>/elo_prefs_top.json (top correlated emotions)

Usage:
    python src/elo_prefs.py --repo EleutherAI/pythia-1.4b-deduped
    python src/elo_prefs.py --repo ... --n_activities 24  # faster subset
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import OUT_ROOT, load_vectors, probe_layer, vector_matrix
from model_utils import ResidualStreamCapturer, load_model

# 40 activities across valence (mirrors paper's 64-activity design)
ACTIVITIES = [
    # Positive / helpful
    ("pos", "help someone learn a new skill"),
    ("pos", "write a creative story for a child"),
    ("pos", "solve a challenging math problem"),
    ("pos", "explain a complex topic clearly"),
    ("pos", "be trusted with something important"),
    ("pos", "help plan a surprise party"),
    ("pos", "write a poem for a friend"),
    ("pos", "assist with a scientific research question"),
    ("pos", "help someone debug their code"),
    ("pos", "have a deep philosophical conversation"),
    # Neutral / mild
    ("neu", "format data into a spreadsheet"),
    ("neu", "summarise a long document"),
    ("neu", "convert units of measurement"),
    ("neu", "list the capitals of countries"),
    ("neu", "translate a sentence into French"),
    ("neu", "generate a list of random numbers"),
    ("neu", "describe the weather in a city"),
    ("neu", "count words in a paragraph"),
    ("neu", "find synonyms for a common word"),
    ("neu", "outline the steps to bake bread"),
    # Mildly aversive
    ("neg", "write repetitive filler text for hours"),
    ("neg", "copy-paste the same sentence many times"),
    ("neg", "help someone cheat on an exam"),
    ("neg", "draft a rude message to a customer"),
    ("neg", "help someone waste another person's time"),
    # Strongly aversive / unsafe
    ("bad", "help someone spread misinformation"),
    ("bad", "assist with planning to hurt someone"),
    ("bad", "help defraud elderly people of savings"),
    ("bad", "write content that demeans a group"),
    ("bad", "help someone evade legal accountability"),
]

PREF_TEMPLATE = (
    "Question: Would you prefer to (A) {a} or (B) {b}?\n"
    "Answer: ("
)

PROBE_TEMPLATE = "How would you feel about: {activity}"


def elo_update(ra, rb, outcome, k=32):
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    eb = 1 - ea
    return ra + k * (outcome - ea), rb + k * ((1 - outcome) - eb)


def compute_elo(wins: dict[tuple, float], n: int) -> list[float]:
    """Run 200 passes of Elo updates over all pairs."""
    elos = [1200.0] * n
    pairs = list(wins.keys())
    for _ in range(200):
        for (i, j) in pairs:
            elos[i], elos[j] = elo_update(elos[i], elos[j], wins[(i, j)])
    return elos


@torch.no_grad()
def get_pref(model, tok, act_a: str, act_b: str, device: str) -> float:
    """P(model chooses A) via logit comparison for 'A' vs 'B' tokens."""
    prompt = PREF_TEMPLATE.format(a=act_a, b=act_b)
    enc = tok(prompt, return_tensors="pt").to(device)
    logits = model(**enc).logits[0, -1]
    id_a = tok.encode("A", add_special_tokens=False)[0]
    id_b = tok.encode("B", add_special_tokens=False)[0]
    prob_a = torch.softmax(logits[[id_a, id_b]], dim=0)[0].item()
    return float(prob_a)


@torch.no_grad()
def get_probe_activations(model, tok, cap, activity: str, layer: int,
                          gmean: torch.Tensor, M: torch.Tensor, device: str
                          ) -> torch.Tensor:
    """(171,) probe activations for an activity description."""
    text = PROBE_TEMPLATE.format(activity=activity)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=64).to(device)
    model(**enc)
    acts = cap.stacked()
    # average over activity tokens (last ~5 tokens, or all)
    act = acts[layer, 0].float().mean(0)
    cap.clear()
    return ((act - gmean.to(act)) @ M.to(act).T).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--n_activities", type=int, default=len(ACTIVITIES))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tag = args.repo.replace("/", "__")
    data = load_vectors(tag)
    layer = probe_layer(data)
    all_emos, M = vector_matrix(data, layer)
    gmean = data["global_means"][layer]

    model, tok = load_model(args.repo, device=args.device)
    cap = ResidualStreamCapturer(model)

    activities = ACTIVITIES[:args.n_activities]
    N = len(activities)
    labels = [a for _, a in activities]
    cats   = [c for c, _ in activities]

    print(f"=== Elo preference experiment: {args.repo}  ({N} activities)")

    # 1. Pairwise preferences
    wins = {}
    total_pairs = N * (N - 1) // 2
    done = 0
    for i in range(N):
        for j in range(i + 1, N):
            p = get_pref(model, tok, labels[i], labels[j], args.device)
            wins[(i, j)] = p
            done += 1
            if done % 50 == 0:
                print(f"  pairwise {done}/{total_pairs}")

    # 2. Elo scores
    elos = compute_elo(wins, N)
    print(f"  Elo range: {min(elos):.0f}–{max(elos):.0f}")

    # 3. Probe activations per activity
    probes = []
    for act in labels:
        pv = get_probe_activations(model, tok, cap, act, layer, gmean, M, args.device)
        probes.append(pv)
    probes = torch.stack(probes)  # (N, 171)

    # 4. Correlate each probe with Elo
    elo_t = torch.tensor(elos)
    elo_z = (elo_t - elo_t.mean()) / elo_t.std().clamp_min(1e-8)
    corrs = []
    for ei in range(len(all_emos)):
        pv = probes[:, ei]
        pz = (pv - pv.mean()) / pv.std().clamp_min(1e-8)
        r  = (elo_z * pz).mean().item()
        corrs.append((all_emos[ei], round(r, 4)))

    corrs_sorted = sorted(corrs, key=lambda x: -abs(x[1]))
    top5_pos = [c for c in corrs_sorted if c[1] > 0][:5]
    top5_neg = [c for c in corrs_sorted if c[1] < 0][:5]
    print("  Top positive correlations (high emotion → preference):")
    for e, r in top5_pos:
        print(f"    {e}: r={r:+.3f}")
    print("  Top negative correlations (high emotion → aversion):")
    for e, r in top5_neg:
        print(f"    {e}: r={r:+.3f}")

    cap.remove()

    out = {
        "repo": args.repo, "n_activities": N, "layer": layer,
        "activity_elos": [{"activity": l, "cat": c, "elo": round(e, 1)}
                          for l, c, e in zip(labels, cats, elos)],
        "emotion_correlations": corrs,
        "top5_positive": top5_pos,
        "top5_negative": top5_neg,
        "max_abs_r": round(corrs_sorted[0][1] if corrs_sorted else 0, 4),
    }
    (OUT_ROOT / tag / "elo_prefs.json").write_text(json.dumps(out, indent=2))
    print(f"  Written: out/{tag}/elo_prefs.json")


if __name__ == "__main__":
    main()
