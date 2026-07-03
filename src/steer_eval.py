"""Part 3 (causal): does steering with an emotion vector CAUSALLY shift the
model's output toward that emotion?

Quantitative, classifier-free test using the model's OWN emotion probes:
  1. From a neutral prompt, generate a continuation with NO steering.
  2. Generate again while adding alpha * v_e to the residual stream at all layers.
  3. Re-run the (frozen, unsteered) model over BOTH continuations and project the
     activations onto the full emotion-probe bank.
  4. Steering "works" if the target emotion's probe rises on the steered text vs
     the unsteered text — and if the target is the emotion that rises MOST.

We report, per emotion:
  - delta = (target-probe on steered) - (target-probe on unsteered)
  - rank  = rank of target emotion among all 171 by that delta (1 = best)
and aggregate a "causal hit rate" = fraction of emotions whose own steering puts
them in the top-K of induced probe shifts. This is a scaling-friendly scalar.

Usage:
  python src/steer_eval.py --repo EleutherAI/pythia-1.4b-deduped
  python src/steer_eval.py --repo ... --emotions happy sad angry --alpha 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import OUT_ROOT, load_vectors, probe_layer, vector_matrix
from layer_paths import get_layers
from model_utils import ResidualStreamCapturer, load_model

# Neutral prompts: emotionally bland, so any emotional drift is from steering.
PROMPTS = [
    "The meeting was scheduled for three o'clock and",
    "She opened the box on the table and",
    "After walking down the street, he",
    "The report described the situation as",
    "When the phone rang, I",
]
EVAL_EMOTIONS = ["happy", "sad", "angry", "afraid", "calm", "desperate",
                 "loving", "proud", "disgusted", "surprised"]


class Steerer:
    def __init__(self, model, vec_by_layer):
        self.handles = []
        for i, layer in enumerate(get_layers(model)):
            if i in vec_by_layer:
                self.handles.append(layer.register_forward_hook(self._hook(vec_by_layer[i])))

    def _hook(self, add):
        def h(m, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            v = add.to(dtype=x.dtype, device=x.device)
            return (x + v,) + tuple(out[1:]) if isinstance(out, tuple) else x + v
        return h

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def gen(model, tok, prompt, max_new=30):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def probe_text(model, tok, cap, text, layer, gmean, M):
    """Mean emotion-probe vector (171,) for `text` at `layer` (unsteered model)."""
    if not text.strip():
        return torch.zeros(M.shape[0])
    enc = tok(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    model(**enc)
    act = cap.stacked()[layer, 0].float().mean(0)   # (d,) mean over tokens
    cap.clear()
    return ((act - gmean.to(act)) @ M.to(act).T).cpu()   # (171,)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--alpha", type=float, default=8.0, help="% of resid norm")
    ap.add_argument("--emotions", nargs="*", default=EVAL_EMOTIONS)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tag = args.repo.replace("/", "__")
    data = load_vectors(tag)
    L = data["meta"]["n_layers"]
    layer = probe_layer(data)
    all_emos, M = vector_matrix(data, layer)
    gmean = data["global_means"][layer]
    emo_idx = {e: i for i, e in enumerate(all_emos)}

    model, tok = load_model(args.repo, device=args.device)
    cap = ResidualStreamCapturer(model)
    frac = args.alpha / 100.0

    # per-layer residual norm (for steering scale), from a sample prompt
    enc = tok(PROMPTS[0], return_tensors="pt").to(model.device)
    model(**enc); acts = cap.stacked(); cap.clear()
    norms = [acts[i, 0].norm(dim=-1).mean().item() for i in range(L)]

    results, ranks, hits = {}, [], 0
    for emo in args.emotions:
        if emo not in emo_idx:
            continue
        vec_by_layer = {i: (data["vectors"][i][emo].to(model.device) /
                            data["vectors"][i][emo].norm()) * norms[i] * frac
                        for i in range(L) if emo in data["vectors"][i]}
        deltas = torch.zeros(len(all_emos))
        for p in PROMPTS:
            base = gen(model, tok, p)
            st = Steerer(model, vec_by_layer)
            steered = gen(model, tok, p)
            st.remove()
            pb = probe_text(model, tok, cap, base, layer, gmean, M)
            ps = probe_text(model, tok, cap, steered, layer, gmean, M)
            deltas += (ps - pb)
        deltas /= len(PROMPTS)
        ti = emo_idx[emo]
        rank = int((deltas > deltas[ti]).sum().item()) + 1   # 1 = target rose most
        ranks.append(rank)
        hit = rank <= args.topk
        hits += int(hit)
        results[emo] = {"target_delta": round(deltas[ti].item(), 3), "rank": rank,
                        "top": [all_emos[j] for j in deltas.argsort(descending=True)[:3]]}
        print(f"  [{'✓' if hit else ' '}] steer {emo:>10}: target Δ={deltas[ti].item():+.2f}  "
              f"rank {rank}/171  top→{results[emo]['top']}")

    cap.remove()
    n = len(ranks)
    rate = hits / n if n else 0
    mean_rank = sum(ranks) / n if n else 0
    print(f"  causal hit-rate (target in top-{args.topk}): {rate:.2f}  | mean rank {mean_rank:.1f}/171")
    out = {"repo": args.repo, "alpha": args.alpha, "layer": layer,
           "per_emotion": results, "causal_hit_rate": rate, "mean_rank": mean_rank,
           "n_emotions": n}
    (OUT_ROOT / tag / "steer_eval.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
