"""Extract per-story mean residual-stream activations for every emotion, at every
layer, for a given model. Works for any family in layer_paths.FAMILIES.

Memory-light: unlike the reference repo (which dumps every token's activation =>
~83 GB/model), we fold the token-averaging (from MIN_TOKEN_OFFSET onward) into
the forward loop and write a single compact file per model:

    out/<model_tag>/story_means.pt   ~  (171 emotions -> tensor (n_stories, L, d))
    out/<model_tag>/neutral_means.pt ~  (n_neutral, L, d)
    out/<model_tag>/meta.json

These per-story means are all compute_vectors.py needs.

Usage:
    python src/extract.py --repo EleutherAI/pythia-70m-deduped
    python src/extract.py --repo cerebras/Cerebras-GPT-111M --batch-size 64
    python src/extract.py --repo Qwen/Qwen3.5-0.8B-Base
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from model_utils import ResidualStreamCapturer, load_model

PROJECT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT / "out"
DATASET_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--ryancodrai--emotion-probes/snapshots/*"
)

MIN_TOKEN_OFFSET = 50   # paper: average tokens from the 50th onward
MIN_USABLE = MIN_TOKEN_OFFSET + 10  # skip stories shorter than this
MAX_LEN = 512


def model_tag(repo: str) -> str:
    return repo.replace("/", "__")


def dataset_file(rel: str) -> str:
    snap = sorted(glob.glob(DATASET_GLOB))
    if not snap:
        raise FileNotFoundError(
            "emotion-probes dataset not found in HF cache. "
            "Run scripts/download_models.sh dataset"
        )
    matches = glob.glob(os.path.join(snap[-1], "**", rel), recursive=True)
    if not matches:
        raise FileNotFoundError(f"{rel} not found under {snap[-1]}")
    return matches[0]


@torch.no_grad()
def mean_acts_for_texts(model, tok, texts: list[str], cap, batch_size: int):
    """Return (kept_indices, tensor(kept, L, d)) of per-text mean activations
    over tokens [MIN_TOKEN_OFFSET : seq_len]. Texts too short are dropped."""
    out_vecs, kept = [], []
    device = model.device
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(batch, return_tensors="pt", truncation=True,
                  max_length=MAX_LEN, padding=True).to(device)
        model(**enc)
        acts = cap.stacked()                 # (L, B, S, d) on GPU
        mask = enc["attention_mask"]         # (B, S)
        for j in range(acts.shape[1]):
            seq_len = int(mask[j].sum().item())
            if seq_len < MIN_USABLE:
                continue
            # Reduce on GPU (cheap), then move only the tiny (L, d) result to CPU.
            v = acts[:, j, MIN_TOKEN_OFFSET:seq_len, :].mean(dim=1).float().cpu()
            out_vecs.append(v)
            kept.append(i + j)
        # CRITICAL on unified-memory GB10: free this batch's GPU activations now,
        # else captures + stacked copies accumulate and OOM the shared 128GB pool
        # (took down VS Code on the 13B model). Drop refs each batch.
        del acts
        cap.clear()
    if not out_vecs:
        return [], torch.empty(0)
    return kept, torch.stack(out_vecs)       # (kept, L, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit-per-emotion", type=int, default=0,
                    help="cap stories per emotion (0 = all 1200); use small for smoke tests")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the forward pass (~20%% faster; lossless; "
                         "~30s one-time warmup). Worth it for slow models like Qwen3.5.")
    ap.add_argument("--keep-ckpt", action="store_true",
                    help="keep per-emotion ckpt/ files after a successful run (default: delete)")
    args = ap.parse_args()

    tag = model_tag(args.repo)
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.repo} ...")
    model, tok = load_model(args.repo, device=args.device)
    if args.compile:
        torch.set_float32_matmul_precision("high")
        model.forward = torch.compile(model.forward, dynamic=True)
        print("  torch.compile enabled (first batch compiles, ~30s)")
    cap = ResidualStreamCapturer(model)
    L, d = cap.n_layers, cap.d_model
    print(f"  {L} layers, d_model={d}, dtype={next(model.parameters()).dtype}")

    # ---- Stories: group by emotion ----
    stories = pd.read_parquet(dataset_file("expression/stories.parquet"))
    if args.limit_per_emotion:
        stories = (stories.groupby("emotion", group_keys=False)
                          .head(args.limit_per_emotion))
    by_emotion: dict[str, list[str]] = defaultdict(list)
    for emo, txt in zip(stories["emotion"], stories["story"]):
        by_emotion[emo].append(txt)
    emotions = sorted(by_emotion)
    print(f"  {len(emotions)} emotions, {sum(len(v) for v in by_emotion.values())} stories")

    # ---- Resumable extraction: checkpoint each emotion to ckpt/<emo>.pt as it
    #      completes. On restart we skip emotions already saved, so a kill
    #      mid-model only loses the in-flight emotion (~13s), not the whole run.
    ckpt_dir = out_dir / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Invalidate checkpoints if extraction params changed (e.g. different N),
    # so a resume never mixes runs with different stories-per-emotion.
    stamp = ckpt_dir / "_params.json"
    cur = {"limit_per_emotion": args.limit_per_emotion,
           "min_token_offset": MIN_TOKEN_OFFSET, "max_len": MAX_LEN}
    if stamp.exists() and json.loads(stamp.read_text()) != cur:
        import shutil
        print("  ckpt params changed -> discarding stale checkpoints")
        shutil.rmtree(ckpt_dir); ckpt_dir.mkdir(parents=True)
    stamp.write_text(json.dumps(cur))

    def ckpt_path(emo):  # filesystem-safe name
        return ckpt_dir / (emo.replace("/", "_").replace(" ", "_") + ".pt")

    done = [e for e in emotions if ckpt_path(e).exists()]
    todo = [e for e in emotions if not ckpt_path(e).exists()]
    if done:
        print(f"  resuming: {len(done)} emotions already checkpointed, {len(todo)} to go")

    t0 = time.time()
    for emo in tqdm(todo, desc="emotions", initial=len(done), total=len(emotions)):
        _, vecs = mean_acts_for_texts(model, tok, by_emotion[emo], cap, args.batch_size)
        # atomic write: tmp then rename, so a kill mid-save can't leave a partial file
        tmp = ckpt_path(emo).with_suffix(".pt.tmp")
        torch.save(vecs, tmp)                # (n_e, L, d)
        tmp.rename(ckpt_path(emo))

    # Assemble full story_means from all checkpoints.
    story_means = {e: torch.load(ckpt_path(e), weights_only=True) for e in emotions}
    torch.save(story_means, out_dir / "story_means.pt")
    print(f"  stories done in {(time.time()-t0)/60:.1f} min -> story_means.pt")

    # ---- Neutral stories (for PCA confound removal) ----  [also resumable]
    neutral_ckpt = ckpt_dir / "_neutral.pt"
    if neutral_ckpt.exists():
        neutral_means = torch.load(neutral_ckpt, weights_only=True)
        print(f"  neutral: resumed from checkpoint  shape={tuple(neutral_means.shape)}")
    else:
        neutral = pd.read_parquet(dataset_file("expression/neutral_stories.parquet"))
        _, neutral_means = mean_acts_for_texts(
            model, tok, list(neutral["story"]), cap, args.batch_size)
        tmp = neutral_ckpt.with_suffix(".pt.tmp")
        torch.save(neutral_means, tmp); tmp.rename(neutral_ckpt)
    torch.save(neutral_means, out_dir / "neutral_means.pt")  # (n, L, d)
    print(f"  neutral done -> neutral_means.pt  shape={tuple(neutral_means.shape)}")

    cap.remove()
    meta = {"repo": args.repo, "tag": tag, "n_layers": L, "d_model": d,
            "n_emotions": len(emotions), "limit_per_emotion": args.limit_per_emotion,
            "min_token_offset": MIN_TOKEN_OFFSET, "max_len": MAX_LEN}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # Full run succeeded -> drop the per-emotion checkpoints (story_means.pt has them all).
    if not args.keep_ckpt:
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
