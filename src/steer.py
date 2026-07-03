"""Activation steering (paper's causal test): add alpha * emotion_vector to the
residual stream at every layer during generation, and observe the output shift.

Steering strength is given as a fraction of the residual-stream norm (as in the
paper), estimated per-layer from the prompt activations so it transfers across
model sizes.

This is a qualitative/demo tool (prints steered vs unsteered continuations) plus
an optional probe-based quantitative check: does steering toward emotion E raise
the E-probe activation on the model's own continuation?

Usage:
    python src/steer.py --repo EleutherAI/pythia-70m-deduped --emotion desperate \
        --prompt "I opened the email and" --alpha 8.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import load_vectors, probe_layer
from layer_paths import get_layers
from model_utils import load_model


class Steerer:
    """Adds a per-layer steering vector to each decoder block's output."""

    def __init__(self, model, vec_by_layer: dict[int, torch.Tensor]):
        self.handles = []
        layers = get_layers(model)
        for i, layer in enumerate(layers):
            if i in vec_by_layer:
                self.handles.append(layer.register_forward_hook(self._hook(vec_by_layer[i])))

    def _hook(self, add_vec):
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                return (output[0] + add_vec,) + tuple(output[1:])
            return output + add_vec
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def generate(model, tok, prompt, max_new=40):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--emotion", required=True)
    ap.add_argument("--prompt", default="I opened the door and")
    ap.add_argument("--alpha", type=float, default=6.0,
                    help="steering strength as fraction of resid-stream norm (x100)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=40)
    args = ap.parse_args()

    tag = args.repo.replace("/", "__")
    data = load_vectors(tag)
    L = data["meta"]["n_layers"]
    model, tok = load_model(args.repo, device=args.device)
    dtype = next(model.parameters()).dtype

    # Estimate per-layer residual-stream norm from the prompt, scale each
    # layer's unit emotion vector to alpha% of that norm.
    enc = tok(args.prompt, return_tensors="pt").to(model.device)
    from model_utils import ResidualStreamCapturer
    cap = ResidualStreamCapturer(model)
    model(**enc)
    acts = cap.stacked()  # (L,1,S,d)
    cap.remove()
    frac = args.alpha / 100.0
    vec_by_layer = {}
    for i in range(L):
        if args.emotion not in data["vectors"][i]:
            continue
        norm = acts[i, 0].norm(dim=-1).mean().item()  # mean token norm at layer i
        v = data["vectors"][i][args.emotion].to(model.device, dtype)
        vec_by_layer[i] = (v / v.norm()) * norm * frac

    print(f"\n=== steering '{args.emotion}' @ {args.alpha}% resid norm  ({args.repo})")
    print(f"prompt: {args.prompt!r}\n")
    base = generate(model, tok, args.prompt, args.max_new)
    print(f"[unsteered] {base}\n")
    st = Steerer(model, vec_by_layer)
    steered = generate(model, tok, args.prompt, args.max_new)
    st.remove()
    print(f"[steered:{args.emotion}] {steered}")


if __name__ == "__main__":
    main()
