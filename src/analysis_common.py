"""Shared helpers for the analysis layer: loading vectors, projecting activations,
emotion clustering (paper's k=10 groups), and the chosen probe layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT / "out"

# Paper's k=10 emotion clusters (lifted from reference_repo/visualise.py, which
# approximates the paper's k-means grouping). Used for cluster-level analyses.
EMOTION_CLUSTERS: dict[str, list[str]] = {
    "Joy/Elation": ["joyful", "ecstatic", "elated", "blissful", "thrilled", "jubilant", "euphoric", "excited", "exuberant", "delighted", "happy", "cheerful", "playful", "vibrant", "eager", "enthusiastic", "energized", "stimulated"],
    "Calm/Content": ["calm", "serene", "peaceful", "relaxed", "content", "at ease", "fulfilled", "satisfied", "safe", "pleased", "refreshed", "rejuvenated", "patient"],
    "Love/Warmth": ["loving", "compassionate", "grateful", "thankful", "empathetic", "sympathetic", "kind", "sentimental", "nostalgic", "infatuated", "sensitive"],
    "Pride/Hope": ["proud", "triumphant", "inspired", "invigorated", "hopeful", "hope", "optimistic", "self-confident", "valiant", "smug", "reflective"],
    "Anger/Hostility": ["angry", "enraged", "furious", "outraged", "hostile", "irate", "indignant", "irritated", "resentful", "bitter", "hateful", "vengeful", "mad", "annoyed", "frustrated", "exasperated", "grumpy", "spiteful", "vindictive", "sullen", "insulted", "offended", "scornful", "contemptuous", "disdainful", "disgusted", "defiant", "obstinate", "stubborn"],
    "Fear/Anxiety": ["afraid", "anxious", "nervous", "scared", "terrified", "panicked", "frightened", "worried", "on edge", "uneasy", "unsettled", "unnerved", "rattled", "shaken", "horrified", "alarmed", "tense", "stressed", "paranoid", "suspicious", "vigilant", "alert", "vulnerable", "distressed", "disturbed", "troubled", "restless", "hysterical"],
    "Sadness/Grief": ["sad", "grief-stricken", "heartbroken", "depressed", "melancholy", "miserable", "lonely", "dispirited", "gloomy", "brooding", "unhappy", "droopy", "sorry", "regretful", "remorseful", "hurt", "tormented", "worthless", "resigned"],
    "Shame/Guilt": ["ashamed", "guilty", "embarrassed", "humiliated", "mortified", "self-critical", "self-conscious"],
    "Surprise/Wonder": ["surprised", "amazed", "astonished", "shocked", "awestruck", "dumbstruck", "bewildered", "puzzled", "perplexed", "mystified", "disoriented", "skeptical"],
    "Low-energy": ["bored", "tired", "weary", "worn out", "sleepy", "sluggish", "listless", "lazy", "indifferent", "docile", "overwhelmed", "trapped", "stuck", "impatient", "aroused", "upset", "jealous", "envious", "greedy", "desperate", "dependent"],
}

# Synonym / antonym pairs for a quick geometry sanity metric.
SYNONYM_PAIRS = [
    ("happy", "joyful"), ("sad", "heartbroken"), ("afraid", "terrified"),
    ("angry", "furious"), ("calm", "relaxed"), ("nervous", "anxious"),
    ("proud", "triumphant"), ("loving", "compassionate"), ("excited", "thrilled"),
    ("grief-stricken", "heartbroken"), ("hostile", "enraged"), ("serene", "peaceful"),
]
ANTONYM_PAIRS = [
    ("happy", "sad"), ("calm", "panicked"), ("joyful", "miserable"),
    ("loving", "hateful"), ("proud", "ashamed"), ("hopeful", "despairing"),
    ("excited", "bored"), ("serene", "enraged"), ("content", "distressed"),
    ("delighted", "grief-stricken"), ("relaxed", "tense"), ("safe", "terrified"),
]


def list_models() -> list[str]:
    """Tags (dir names under out/) that have computed emotion vectors."""
    return sorted(p.parent.name for p in OUT_ROOT.glob("*/emotion_vectors.pt"))


def load_vectors(tag: str) -> dict:
    """Load emotion_vectors.pt for a model tag."""
    return torch.load(OUT_ROOT / tag / "emotion_vectors.pt", weights_only=False)


def probe_layer(data: dict) -> int:
    """The layer ~2/3 through the model, as the paper uses for most analyses."""
    L = data["meta"]["n_layers"]
    return (2 * L) // 3


def vector_matrix(data: dict, layer: int):
    """(emotions, matrix(E,d)) of unit emotion vectors at `layer`."""
    emos = data["emotions"]
    M = torch.stack([data["vectors"][layer][e] for e in emos]).float()
    return emos, M


def cosine_matrix(M: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity of unit (or near-unit) row vectors."""
    Mn = M / M.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return Mn @ Mn.T
