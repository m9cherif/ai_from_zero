"""Score a checkpoint on a held-out corpus.

    python scripts/evaluate.py --data data/holdout
    python scripts/evaluate.py --checkpoint output/checkpoints/checkpoint_best.pt --data data/val

Reports token-weighted perplexity over every full window of the corpus, plus
sample generations. Perplexity is only meaningful on text the model has never
seen - use scripts/fetch_corpus.py's leakage check if you are unsure.
"""

import argparse
import glob
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from myai.data.streaming import PretokenizedDataset, TokenCache
from myai.nn.model import LanguageModel
from myai.inference import InferenceEngine
from myai.tokenizer import load_tokenizer

PROMPTS = [
    "To be, or not",
    "The king",
    "It was a",
    "She looked at him and",
    "In the beginning",
]


def find_latest_checkpoint(directory: str) -> str:
    latest = os.path.join(directory, "checkpoint_latest.pt")
    if os.path.exists(latest):
        return latest

    def step_of(path: str) -> int:
        match = re.search(r"checkpoint_step_(\d+)\.pt$", path.replace("\\", "/"))
        return int(match.group(1)) if match else -1

    candidates = sorted(glob.glob(os.path.join(directory, "checkpoint_step_*.pt")), key=step_of)
    return candidates[-1] if candidates else ""


@torch.no_grad()
def perplexity(model: LanguageModel, dataset, batch_size: int, max_batches: int) -> tuple:
    """Token-weighted mean loss over the corpus."""
    model.eval()
    device = model.device

    total_loss = 0.0
    total_tokens = 0
    batches = 0
    buffer = []

    def flush(seqs):
        nonlocal total_loss, total_tokens, batches
        ids = torch.tensor(seqs, device=device)
        out = model(ids, labels=ids)
        # Each window contributes seq_len - 1 predicted tokens.
        n = ids.numel() - ids.shape[0]
        total_loss += float(out["loss"].item()) * n
        total_tokens += n
        batches += 1

    for window in dataset:
        buffer.append(window)
        if len(buffer) == batch_size:
            flush(buffer)
            buffer = []
            if max_batches and batches >= max_batches:
                break
    if buffer and (not max_batches or batches < max_batches):
        flush(buffer)

    mean = total_loss / max(total_tokens, 1)
    return mean, math.exp(min(mean, 20)), total_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Defaults to the newest in --checkpoint-dir")
    parser.add_argument("--checkpoint-dir", default="output/checkpoints")
    parser.add_argument("--tokenizer", default="output/tokenizer.json")
    parser.add_argument("--data", nargs="+", required=True, help="Held-out corpus directories or files")
    parser.add_argument("--cache", default=None, help="Where to write the token cache")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=0, help="0 scores the whole corpus")
    parser.add_argument("--seq-len", type=int, default=None, help="Defaults to the model's context")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-samples", action="store_true")
    args = parser.parse_args()

    checkpoint = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)
    if not checkpoint or not os.path.exists(checkpoint):
        print(f"No checkpoint found (looked in {args.checkpoint_dir})", file=sys.stderr)
        sys.exit(1)

    tokenizer = load_tokenizer(args.tokenizer)
    engine = InferenceEngine(device=args.device)
    engine.load_checkpoint(checkpoint, tokenizer_path=args.tokenizer)
    model = engine.model

    seq_len = args.seq_len or model.config.max_seq_len
    cache_path = args.cache or os.path.join(
        os.path.dirname(checkpoint) or ".", "_eval_tokens.bin"
    )
    TokenCache.build(args.data, tokenizer, cache_path, overwrite=True)
    dataset = PretokenizedDataset(cache_path, max_seq_len=seq_len, shuffle=False)

    print(f"\ncheckpoint  {checkpoint}")
    print(f"parameters  {model.num_parameters():,}")
    print(f"corpus      {dataset.n_tokens:,} tokens, {dataset.n_windows:,} windows of {seq_len}")

    start = time.time()
    loss, ppl, scored = perplexity(model, dataset, args.batch_size, args.max_batches)
    elapsed = time.time() - start

    baseline = math.log(tokenizer.vocab_size)
    print(f"\n  loss        {loss:.4f}   (uniform baseline {baseline:.4f})")
    print(f"  perplexity  {ppl:.2f}     (uniform baseline {tokenizer.vocab_size})")
    print(f"  scored      {scored:,} tokens in {elapsed:.0f}s")

    if not args.no_samples:
        print("\nSamples:")
        for prompt in PROMPTS:
            text = engine.generate(
                prompt, max_new_tokens=60, temperature=0.8,
                top_k=40, min_p=0.05, repetition_penalty=1.15,
            )
            continuation = text[len(prompt):].strip().replace("\n", " / ")
            print(f"  {prompt!r} -> {continuation[:110]!r}")


if __name__ == "__main__":
    main()
