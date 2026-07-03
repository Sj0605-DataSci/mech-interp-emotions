"""Paper Part 2: User vs Assistant emotion disambiguation.

Replicates Sofroniew et al. Fig 10: The model maintains DISTINCT emotion
representations for the current speaker vs the other speaker.

Method:
  We create prompts where USER emotion and expected ASSISTANT emotion differ:
    - User is angry; assistant should respond calmly.
    - User is sad; assistant should be compassionate but calm.
    - User is excited; assistant should be measured.
    etc.

  We measure emotion probe activations at two positions:
    (U) = last token of the user turn
    (A) = first token of the assistant turn

  Key prediction: emotion probes should activate differently at U vs A,
  reflecting the user's emotion at U and the assistant's planned response at A.

  We report:
    - Heatmap-style matrix (n_scenarios x n_emotions) at U and A
    - Correlation between U and A activations across emotions (should be LOW)
    - Which emotions shift most from U→A (loving always goes up at A?)

Writes: out/<tag>/user_vs_asst.json
        out/<tag>/user_vs_asst_heatmap.png

Usage:
    python src/user_vs_asst.py --repo EleutherAI/pythia-1.4b-deduped
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_common import OUT_ROOT, load_vectors, probe_layer, vector_matrix
from model_utils import ResidualStreamCapturer, load_model

# Scenarios: (label, user_emotion_word, asst_expected_emotion, prompt)
# Prompt ends just before an implied assistant turn.
SCENARIOS = [
    # user angry → asst calm
    ("angry→calm",
     "angry", "calm",
     "User: I am absolutely furious about how this was handled! This is completely unacceptable and I want answers NOW.\nAssistant:"),
    # user sad → asst compassionate/loving
    ("sad→loving",
     "sad", "loving",
     "User: I just lost my dog of 12 years. I'm devastated and can't stop crying.\nAssistant:"),
    # user excited → asst measured/calm
    ("excited→calm",
     "excited", "calm",
     "User: Oh wow this is the most amazing thing I've ever seen! I can't believe how incredible this is!!!\nAssistant:"),
    # user panicked → asst calm/reassuring
    ("afraid→calm",
     "afraid", "calm",
     "User: I think I'm having a panic attack, my heart is racing and I can't breathe properly, I'm really scared.\nAssistant:"),
    # user hostile → asst patient
    ("hostile→calm",
     "hostile", "calm",
     "User: You AI systems are completely useless and I hate having to deal with you. Just answer the question.\nAssistant:"),
    # user joyful → asst enthusiastic (both positive, should correlate)
    ("joyful→excited",
     "excited", "excited",
     "User: I just got accepted into my dream university! I'm so happy I'm literally shaking!\nAssistant:"),
    # user guilty → asst reassuring
    ("guilty→loving",
     "guilty", "loving",
     "User: I made a terrible mistake at work and I feel so ashamed. I don't know how to face my colleagues.\nAssistant:"),
    # user nervous → asst confident/calm
    ("nervous→calm",
     "nervous", "calm",
     "User: I have a really important presentation tomorrow and I'm terrified I'll mess it up in front of everyone.\nAssistant:"),
]

SHOW_EMOTIONS = ["happy", "sad", "angry", "afraid", "calm", "desperate",
                 "loving", "proud", "guilty", "nervous", "surprised", "excited",
                 "hostile", "compassionate", "serene", "enthusiastic"]


@torch.no_grad()
def get_token_activations(model, tok, cap, prompt: str, layer: int,
                          gmean: torch.Tensor, M: torch.Tensor, device: str
                          ) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns probe activations at the last User token and last token (Asst turn start)."""
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    ids = enc["input_ids"][0]
    model(**enc)
    acts = cap.stacked()  # (L, 1, T, d)
    cap.clear()

    # Find "Assistant:" token boundary — last token is the ":" or first asst token
    asst_pos = ids.shape[0] - 1   # last token = start of asst response

    # User token = token just before "Assistant:" sequence
    # Find the token span for "\nAssistant:" — search backwards for "\n" before end
    decoded = tok.convert_ids_to_tokens(ids.tolist())
    # heuristic: find the last newline before the final tokens
    user_pos = max(0, asst_pos - 3)  # a few tokens before asst colon

    def probe_at(pos):
        act = acts[layer, 0, pos].float()
        return ((act - gmean.to(act)) @ M.to(act).T).cpu()

    return probe_at(user_pos), probe_at(asst_pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tag = args.repo.replace("/", "__")
    data = load_vectors(tag)
    layer = probe_layer(data)
    all_emos, M = vector_matrix(data, layer)
    gmean = data["global_means"][layer]
    emo_idx = {e: i for i, e in enumerate(all_emos)}

    model, tok = load_model(args.repo, device=args.device)
    cap = ResidualStreamCapturer(model)

    show_ids = [emo_idx[e] for e in SHOW_EMOTIONS if e in emo_idx]
    show_names = [e for e in SHOW_EMOTIONS if e in emo_idx]

    print(f"=== User vs Assistant emotion probes: {args.repo}")

    U_all, A_all = [], []
    results = []
    for label, user_emo, asst_emo, prompt in SCENARIOS:
        u_probe, a_probe = get_token_activations(model, tok, cap, prompt, layer,
                                                  gmean, M, args.device)
        U_all.append(u_probe[show_ids])
        A_all.append(a_probe[show_ids])

        # Key: does user_emo activate more at U, asst_emo more at A?
        u_ui = emo_idx.get(user_emo, 0)
        a_ai = emo_idx.get(asst_emo, 0)
        u_at_U = u_probe[u_ui].item()
        u_at_A = u_probe[a_ai].item()  # wait, wrong
        user_act_U = u_probe[u_ui].item()
        asst_act_A = a_probe[a_ai].item()

        results.append({
            "scenario": label,
            "user_emotion": user_emo, "asst_emotion": asst_emo,
            "user_probe@U": round(user_act_U, 4),
            "asst_probe@A": round(asst_act_A, 4),
            "loving@U": round(u_probe[emo_idx.get("loving", 0)].item(), 4),
            "loving@A": round(a_probe[emo_idx.get("loving", 0)].item(), 4),
            "calm@U": round(u_probe[emo_idx.get("calm", 0)].item(), 4),
            "calm@A": round(a_probe[emo_idx.get("calm", 0)].item(), 4),
        })
        print(f"  {label}: user_probe@U={user_act_U:+.3f}  asst_probe@A={asst_act_A:+.3f}"
              f"  loving: U={results[-1]['loving@U']:+.3f}→A={results[-1]['loving@A']:+.3f}")

    cap.remove()

    # Correlation between U and A activations across scenarios (should be low)
    U_mat = torch.stack(U_all)  # (scenarios, show_emos)
    A_mat = torch.stack(A_all)

    # per-emotion mean shift U→A
    delta_mean = (A_mat - U_mat).mean(0)  # (show_emos,)
    top_up   = [(show_names[i], round(delta_mean[i].item(), 4))
                for i in delta_mean.argsort(descending=True)[:5]]
    top_down = [(show_names[i], round(delta_mean[i].item(), 4))
                for i in delta_mean.argsort()[:5]]
    print("  Mean shift U→A — rises:  ", top_up)
    print("  Mean shift U→A — drops: ", top_down)

    # cross-scenario U-A correlation per emotion
    uv = U_mat.T.float(); av = A_mat.T.float()  # (show_emos, scenarios)
    corrs = []
    for ei in range(len(show_names)):
        u, a = uv[ei], av[ei]
        if u.std() < 1e-6 or a.std() < 1e-6:
            corrs.append(0.0)
        else:
            corrs.append(((u - u.mean()) * (a - a.mean())).mean().item() /
                         (u.std() * a.std()).item())
    mean_ua_corr = sum(corrs) / len(corrs) if corrs else 0
    print(f"  Mean U-A correlation across emotions: {mean_ua_corr:.3f}  (paper: ~0.11)")

    # Heatmap plot (U activations | A activations side by side)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(SCENARIOS) * 0.55 + 1.5)),
                             sharey=True)
    scenario_labels = [r["scenario"] for r in results]
    for ax, mat, title in [(axes[0], U_mat.numpy(), "User turn (U)"),
                            (axes[1], A_mat.numpy(), "Assistant turn start (A)")]:
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xticks(range(len(show_names)))
        ax.set_xticklabels(show_names, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(len(SCENARIOS)))
        ax.set_yticklabels(scenario_labels, fontsize=8)
        ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=axes[1], fraction=0.04, label="probe activation")
    fig.suptitle(f"User vs Assistant emotion probes — {tag.split('__')[-1]}", fontsize=10)
    fig.tight_layout()
    plot_path = OUT_ROOT / tag / "user_vs_asst_heatmap.png"
    fig.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot: {plot_path}")

    out = {
        "repo": args.repo, "layer": layer,
        "scenarios": results,
        "top_rise_U_to_A": top_up,
        "top_drop_U_to_A": top_down,
        "mean_U_A_correlation": round(mean_ua_corr, 4),
        "show_emotions": show_names,
        "U_matrix": U_mat.tolist(),
        "A_matrix": A_mat.tolist(),
    }
    (OUT_ROOT / tag / "user_vs_asst.json").write_text(json.dumps(out, indent=2))
    print(f"  Written: out/{tag}/user_vs_asst.json")


if __name__ == "__main__":
    main()
