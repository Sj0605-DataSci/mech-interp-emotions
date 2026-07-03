"""Logit-lens analysis (paper Table 1): push each emotion vector through the
model's final-norm + unembedding and read off the top up/down-weighted tokens.

A faithful emotion vector should upweight tokens semantically related to its
emotion (e.g. desperate -> "urgent", "bankrupt"). We also compute a quantitative
"self-token hit" score: does the emotion's own word (and close variants) appear
among the top-K upweighted tokens? This gives a single scaling-friendly number.

Usage:
    python src/analyze_logitlens.py --repo EleutherAI/pythia-70m-deduped
    python src/analyze_logitlens.py --repo ... --emotions desperate happy sad angry
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import OUT_ROOT, load_vectors, probe_layer
from layer_paths import get_final_norm, get_lm_head
from model_utils import load_model

SHOW = ["happy", "sad", "angry", "afraid", "calm", "desperate", "proud",
        "loving", "guilty", "nervous", "surprised", "inspired"]
TOPK = 8


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--emotions", nargs="*", default=None)
    ap.add_argument("--topk", type=int, default=TOPK)
    args = ap.parse_args()

    tag = args.repo.replace("/", "__")
    data = load_vectors(tag)
    layer = probe_layer(data)
    emotions = args.emotions or [e for e in SHOW if e in data["vectors"][layer]]

    model, tok = load_model(args.repo, device=args.device)
    norm = get_final_norm(model)
    head = get_lm_head(model)
    W = head.weight  # (vocab, d)
    device, dtype = W.device, W.dtype

    def decode(ids):
        return [tok.decode([i]).strip() for i in ids]

    hits = 0
    report = {}
    print(f"\n=== logit-lens: {args.repo}  (layer {layer})")
    for emo in emotions:
        v = data["vectors"][layer][emo].to(device=device, dtype=dtype)
        # final norm then unembed (logit lens)
        logits = head(norm(v.unsqueeze(0))).squeeze(0).float()
        up = torch.topk(logits, args.topk).indices.tolist()
        down = torch.topk(-logits, args.topk).indices.tolist()
        up_toks, down_toks = decode(up), decode(down)
        # self-token hit: does emotion word stem appear in top-up tokens?
        stem = emo.replace("-", " ").split()[0][:4].lower()
        hit = any(stem and stem in t.lower() for t in up_toks)
        hits += int(hit)
        report[emo] = {"up": up_toks, "down": down_toks, "self_hit": hit}
        mark = "✓" if hit else " "
        print(f"  [{mark}] {emo:>12} ↑ {', '.join(t for t in up_toks if t)[:60]}")

    score = hits / max(1, len(emotions))
    report["_self_hit_rate"] = round(score, 4)
    report["_n_emotions"] = len(emotions)
    print(f"  self-token hit rate: {score:.2f} ({hits}/{len(emotions)})")
    (OUT_ROOT / tag / "logitlens.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
