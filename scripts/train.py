"""Train a language model on the corpus in data/.

    python scripts/train.py --preset tiny --steps 2000
    python scripts/train.py --preset small --steps 20000 --batch-size 16

Uses the full library stack: config -> Trainer -> StreamingDataset, with
checkpointing, evaluation and resume.
"""

import argparse
import glob
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from myai.config.presets import MODEL_PRESETS, TrainConfig
from myai.data.streaming import PretokenizedDataset, StreamingDataset, TokenCache
from myai.tokenizer import load_tokenizer
from myai.train.engine import Trainer


def build_config(args, vocab_size: int) -> TrainConfig:
    model = dict(MODEL_PRESETS[args.preset])
    model["vocab_size"] = vocab_size
    model["dropout"] = args.dropout
    if args.seq_len:
        model["max_seq_len"] = args.seq_len
    if args.n_layers:
        model["n_layers"] = args.n_layers
    if args.d_model:
        model["d_model"] = args.d_model
        model["d_ff"] = args.d_model * 4

    return TrainConfig(
        model=model,
        data={
            "data_paths": args.data,
            "batch_size": args.batch_size,
            "max_seq_len": model["max_seq_len"],
            "shuffle_buffer_size": 2000,
            "pack_sequences": True,
        },
        optimizer={
            "optimizer": "adamw",
            "learning_rate": args.lr,
            "weight_decay": 0.1,
            "beta1": 0.9,
            "beta2": 0.95,
            "max_grad_norm": 1.0,
            "gradient_accumulation_steps": args.grad_accum,
        },
        scheduler={
            "scheduler": "warmup_cosine",
            "warmup_steps": max(args.steps // 20, 10),
            "min_lr_ratio": 0.1,
        },
        checkpoint={
            "save_dir": args.output,
            "save_every_steps": args.save_every or max(args.steps // 4, 100),
            "keep_last_n": 3,
        },
        logging={"log_every_steps": args.log_every},
        hardware={
            "device": args.device,
            "dtype": args.dtype,
            "use_mixed_precision": args.mixed_precision,
            "compile_model": args.compile,
        },
        max_steps=args.steps,
        num_epochs=args.epochs,
        # Floored at 1, not 100: a short run with --val-data would otherwise
        # never evaluate and never write a best checkpoint.
        eval_every_steps=max(args.steps // 4, 1),
        eval_steps=20,
        seed=args.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="tiny")
    parser.add_argument("--data", nargs="+", default=["data"])
    parser.add_argument(
        "--val-data", nargs="+", default=None,
        help="Held-out paths. Without these no evaluation runs and no "
             "checkpoint_best.pt is written.",
    )
    parser.add_argument("--tokenizer", default="output/tokenizer.json")
    parser.add_argument("--output", default="output/checkpoints")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None, help="Override the preset's depth")
    parser.add_argument("--d-model", type=int, default=None, help="Override the preset's width")
    parser.add_argument("--save-every", type=int, default=None, help="Checkpoint interval in steps")
    parser.add_argument(
        "--cache-dir", default=None,
        help="Tokenize the corpus once into this directory and train from the "
             "cache. Removes per-epoch tokenization, which is a large share of "
             "CPU step time.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--flash", action="store_true", help="Use fused SDPA attention")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.tokenizer):
        print(
            f"Tokenizer not found at {args.tokenizer}.\n"
            f"Build one first:  python scripts/build_tokenizer.py --type char",
            file=sys.stderr,
        )
        sys.exit(1)

    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Tokenizer: {type(tokenizer).__name__}, vocab={tokenizer.vocab_size}")

    config = build_config(args, tokenizer.vocab_size)
    config.model.use_flash = args.flash
    if args.resume:
        config.checkpoint.resume_from = args.resume

    def build_dataset(paths, name):
        if args.cache_dir:
            cache = TokenCache.build(
                paths, tokenizer, os.path.join(args.cache_dir, f"{name}.bin")
            )
            return PretokenizedDataset(
                cache, max_seq_len=config.data.max_seq_len, seed=config.seed
            )
        stream = StreamingDataset(
            data_paths=paths,
            tokenizer=tokenizer,
            max_seq_len=config.data.max_seq_len,
            shuffle_buffer_size=config.data.shuffle_buffer_size,
            pack_sequences=True,
            seed=config.seed,
        )
        stream.discover_files()
        return stream

    dataset = build_dataset(config.data.data_paths, "train")
    if args.cache_dir:
        print(f"Train corpus: {dataset.n_tokens:,} tokens, "
              f"{dataset.n_windows:,} windows of {config.data.max_seq_len}")

    val_dataset = None
    if args.val_data:
        val_dataset = build_dataset(args.val_data, "val")
        print(f"Validation every {config.eval_every_steps} steps; "
              f"best model tracked in {os.path.join(args.output, 'checkpoint_best.pt')}")

    trainer = Trainer(config, tokenizer=tokenizer)

    start = time.time()
    stats = trainer.train(dataset, val_dataset)
    elapsed = time.time() - start

    vocab = tokenizer.vocab_size
    print()
    print("=" * 60)
    print(f"  steps:      {trainer._loop.global_step}")
    print(f"  time:       {elapsed:.0f}s")
    print(f"  loss:       {stats.get('loss', float('nan')):.4f} "
          f"(random baseline {math.log(vocab):.4f})")
    print(f"  perplexity: {stats.get('perplexity', float('nan')):.2f} "
          f"(random baseline {vocab})")
    print(f"  throughput: {stats.get('tokens_per_second', 0):.0f} tokens/s")
    print("=" * 60)

    print("\nSample generations:")
    model = trainer.model
    for prompt in ["To be, or not", "The king", "It was a"]:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        out = model.generate(
            torch.tensor([ids], device=model.device),
            max_new_tokens=60,
            temperature=0.7,
            top_k=40,
            min_p=0.05,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
        )
        text = tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
        print(f"  {prompt!r} -> {text[len(prompt):].strip()[:70]!r}")


if __name__ == "__main__":
    main()
