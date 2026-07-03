"""Per-family module-path registry for residual-stream hooking.

The emotion-vector method only needs three things from any decoder LM:
  1. the list of decoder blocks (so we can hook each block's residual-stream output),
  2. the final norm + unembedding (for the logit-lens validation),
  3. basic dims (n_layers, d_model) — which we read off the *instantiated* model
     rather than the config, because some new configs (e.g. qwen3_5) nest these.

Every model we study reduces to one of three HF `model_type`s:
  gpt_neox  -> Pythia
  gpt2      -> Cerebras-GPT
  qwen3_5   -> Qwen3.5

Paths verified empirically against:
  EleutherAI/pythia-70m-deduped, cerebras/Cerebras-GPT-111M, Qwen/Qwen3.5-0.8B-Base
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FamilySpec:
    """Dotted attribute paths into an HF CausalLM for one architecture family."""
    model_type: str
    layers_path: str       # ModuleList of decoder blocks
    final_norm_path: str   # final norm before unembed
    lm_head_path: str      # unembedding Linear (or tied embedding)


# Keyed by HF config.model_type.
FAMILIES: dict[str, FamilySpec] = {
    "gpt_neox": FamilySpec("gpt_neox", "gpt_neox.layers", "gpt_neox.final_layer_norm", "embed_out"),
    "gpt2":     FamilySpec("gpt2",     "transformer.h",   "transformer.ln_f",          "lm_head"),
    "qwen3_5":  FamilySpec("qwen3_5",  "model.layers",    "model.norm",                "lm_head"),
    # The loaded CausalLM reports model_type "qwen3_5_text" (its text sub-config),
    # even though the standalone AutoConfig reports "qwen3_5". Same module layout.
    "qwen3_5_text": FamilySpec("qwen3_5_text", "model.layers", "model.norm",          "lm_head"),
    # Forward-compat: if Qwen ships a Qwen3 fallback path, it shares the layout.
    "qwen3":    FamilySpec("qwen3",    "model.layers",    "model.norm",                "lm_head"),
    "qwen3_text": FamilySpec("qwen3_text", "model.layers", "model.norm",              "lm_head"),
}


def _getattr_path(obj, path: str):
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def spec_for(model) -> FamilySpec:
    """Return the FamilySpec for a loaded HF model, by its config.model_type."""
    mt = model.config.model_type
    if mt not in FAMILIES:
        raise KeyError(
            f"Unknown model_type {mt!r}. Add a FamilySpec in layer_paths.py. "
            f"Inspect with: [n for n,m in model.named_modules() if isinstance(m, torch.nn.ModuleList)]"
        )
    return FAMILIES[mt]


def get_layers(model):
    """The ModuleList of decoder blocks. Hook each block's forward output to read
    that block's residual-stream contribution (the standard 'resid_post' point)."""
    return _getattr_path(model, spec_for(model).layers_path)


def get_final_norm(model):
    return _getattr_path(model, spec_for(model).final_norm_path)


def get_lm_head(model):
    return _getattr_path(model, spec_for(model).lm_head_path)


def model_dims(model) -> tuple[int, int]:
    """(n_layers, d_model) read from the instantiated model — robust to nested configs."""
    layers = get_layers(model)
    n_layers = len(layers)
    # d_model: pull from the final norm's weight, which is always (d_model,).
    norm = get_final_norm(model)
    d_model = norm.weight.shape[0]
    return n_layers, d_model
