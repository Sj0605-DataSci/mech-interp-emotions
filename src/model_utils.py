"""Model loading + a forward-hook harness that captures residual-stream
activations at every decoder layer in a single forward pass.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from layer_paths import get_layers, model_dims

# Free, lossless GB10 speedups (~7% on extraction): let cuDNN autotune kernels
# for our repeated batch shapes, and allow TF32 on the fp32 matmul paths.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_model(repo: str, dtype=torch.bfloat16, device: str = "cuda"):
    """Load an HF causal LM + tokenizer, eval mode, on `device`."""
    tok = AutoTokenizer.from_pretrained(repo)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Left-padding is irrelevant here (we mask per-sequence), but right-padding
    # keeps token offsets aligned with the start of each story.
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=dtype, device_map=device)
    model.eval()
    return model, tok


class ResidualStreamCapturer:
    """Registers forward hooks on every decoder block and captures each block's
    output (residual stream after that block) for the most recent forward pass.

    Usage:
        cap = ResidualStreamCapturer(model)
        with torch.no_grad(): model(**inputs)
        acts = cap.stacked()           # (n_layers, batch, seq, d_model)
        cap.remove()
    """

    def __init__(self, model):
        self.model = model
        self.n_layers, self.d_model = model_dims(model)
        self._captured: dict[int, torch.Tensor] = {}
        self._handles = []
        layers = get_layers(model)
        for i, layer in enumerate(layers):
            self._handles.append(layer.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, idx: int):
        def hook(module, inputs, output):
            # Decoder blocks return either a Tensor or a tuple whose first elem
            # is the hidden state. True for gpt_neox, gpt2, and qwen3_5.
            act = output[0] if isinstance(output, tuple) else output
            self._captured[idx] = act.detach()
        return hook

    def stacked(self) -> torch.Tensor:
        """(n_layers, batch, seq, d_model) on the model's device."""
        return torch.stack([self._captured[i] for i in range(self.n_layers)], dim=0)

    def clear(self):
        """Drop captured GPU tensors. Call every batch on unified-memory GB10 to
        avoid accumulating activations in the shared CPU/GPU pool (OOM risk)."""
        self._captured.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
