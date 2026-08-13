"""Measure the effect of each optimization.

    python scripts/benchmark.py
    python scripts/benchmark.py --preset small --steps 20

Reports training step time, generation throughput with and without the KV
cache, and the memory/speed trade-off of gradient checkpointing.
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from myai.config.presets import MODEL_PRESETS
from myai.nn.model import LanguageModel, LMConfig
from myai.train.optimizer import AdamW, build_param_groups


def build(preset: str, vocab: int, **overrides) -> LanguageModel:
    config = dict(MODEL_PRESETS[preset])
    config.update(vocab_size=vocab, dropout=0.0)
    config.update(overrides)
    torch.manual_seed(0)
    return LanguageModel(LMConfig(**config))


def time_it(fn, repeats: int, warmup: int = 2) -> float:
    """Median seconds per call, ignoring warmup iterations."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def bench_training(model, batch, vocab, steps):
    optimizer = AdamW(build_param_groups(model, 0.1), lr=1e-4)
    model.train()

    def step():
        optimizer.zero_grad()
        loss = model(batch, labels=batch)["loss"]
        loss.backward()
        optimizer.step()

    return time_it(step, steps)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="tiny")
    parser.add_argument("--vocab", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--new-tokens", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build(args.preset, args.vocab).to(device)
    seq_len = model.config.max_seq_len

    print("=" * 66)
    print(f"  {model}")
    print(f"  device={device}, batch={args.batch_size}, seq_len={seq_len}")
    print("=" * 66)

    batch = torch.randint(0, args.vocab, (args.batch_size, seq_len), device=device)
    tokens_per_step = args.batch_size * seq_len

    # --- Training step ---------------------------------------------------
    base = bench_training(model, batch, args.vocab, args.steps)
    print(f"\nTraining step")
    print(f"  manual attention      {base * 1000:8.1f} ms   "
          f"({tokens_per_step / base:,.0f} tokens/s)")

    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        model.set_flash_attention(True)
        flash = bench_training(model, batch, args.vocab, args.steps)
        model.set_flash_attention(False)
        print(f"  fused SDPA            {flash * 1000:8.1f} ms   "
              f"({base / flash:.2f}x)")

    model.enable_gradient_checkpointing(True)
    checkpointed = bench_training(model, batch, args.vocab, args.steps)
    model.enable_gradient_checkpointing(False)
    print(f"  gradient checkpoint   {checkpointed * 1000:8.1f} ms   "
          f"({base / checkpointed:.2f}x, saves activation memory)")

    # --- Generation ------------------------------------------------------
    model.eval()
    prompt = torch.randint(0, args.vocab, (1, 16), device=device)

    cached = time_it(
        lambda: model.generate(prompt, max_new_tokens=args.new_tokens,
                               temperature=0.8, top_k=40, use_cache=True),
        repeats=3, warmup=1,
    )
    uncached = time_it(
        lambda: model.generate(prompt, max_new_tokens=args.new_tokens,
                               temperature=0.8, top_k=40, use_cache=False),
        repeats=3, warmup=1,
    )

    print(f"\nGeneration ({args.new_tokens} tokens)")
    print(f"  with KV cache         {cached * 1000:8.1f} ms   "
          f"({args.new_tokens / cached:,.0f} tokens/s)")
    print(f"  without KV cache      {uncached * 1000:8.1f} ms   "
          f"({args.new_tokens / uncached:,.0f} tokens/s)")
    print(f"  speedup               {uncached / cached:8.2f}x")

    # --- Attention shape -------------------------------------------------
    n_kv = model.config.n_kv_heads or model.config.n_heads
    cache_ratio = model.config.n_heads / n_kv
    print(f"\nMemory")
    print(f"  KV cache heads        {n_kv} of {model.config.n_heads} query heads "
          f"({cache_ratio:.0f}x smaller cache than MHA)")
    print(f"  parameters            {model.num_parameters():,} "
          f"({model.num_parameters_excluding_embeddings():,} non-embedding)")
    print()


if __name__ == "__main__":
    main()
