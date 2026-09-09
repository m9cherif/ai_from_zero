"""Talk to your model.

    python scripts/chat.py
    python scripts/chat.py --checkpoint output/checkpoints/checkpoint_latest.pt

Streams tokens as they are generated, using the KV cache.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myai.core.logging import logger, LogLevel
from myai.inference import InferenceEngine
from myai.tokenizer import load_tokenizer


def find_latest_checkpoint(directory: str):
    latest = os.path.join(directory, "checkpoint_latest.pt")
    if os.path.exists(latest):
        return latest
    candidates = glob.glob(os.path.join(directory, "checkpoint_step_*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: os.path.getmtime(p))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--checkpoint-dir", default="output/checkpoints")
    parser.add_argument("--tokenizer", default="output/tokenizer.json")
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--min-p", type=float, default=0.05)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--prompt", nargs="+", default=None,
        help="Answer these prompts and exit instead of starting the REPL. "
             "Use when stdin is not a terminal - scripts, pipes, CI, or a "
             "headless server reached over an agent session.",
    )
    parser.add_argument("--quiet", action="store_true", help="Output only the continuation")
    args = parser.parse_args()

    logger.set_level(LogLevel.WARNING)  # keep the chat output clean

    checkpoint = args.checkpoint or find_latest_checkpoint(args.checkpoint_dir)
    if not checkpoint or not os.path.exists(checkpoint):
        print(
            f"No checkpoint found in {args.checkpoint_dir}.\n"
            f"Train one first:  python scripts/train.py --d-model 320 --n-layers 6 --steps 6500",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(args.tokenizer):
        print(f"Tokenizer not found at {args.tokenizer}", file=sys.stderr)
        sys.exit(1)

    engine = InferenceEngine(device=args.device)
    engine.load_checkpoint(checkpoint, args.tokenizer)

    model = engine.model

    def answer(prompt: str) -> str:
        return engine.generate(
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
        )[len(prompt):].strip()

    if args.prompt:
        for prompt in args.prompt:
            if args.quiet:
                print(answer(prompt))
            else:
                print(f"\nYou: {prompt}\nAI:  {answer(prompt)}")
        return

    print("=" * 60)
    print("  TALK TO YOUR AI - BUILT FROM SCRATCH")
    print("=" * 60)
    print(f"  checkpoint: {checkpoint}")
    print(f"  parameters: {model.num_parameters():,}")
    print(f"  context:    {model.config.max_seq_len} tokens")
    print(f"  attention:  {model.config.n_heads} heads / "
          f"{model.config.n_kv_heads or model.config.n_heads} kv-heads, "
          f"{model.config.position_encoding}")
    print()
    print("  Type a prompt and the model continues it. 'quit' to exit.")
    print()

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            break

        print("AI:  ", end="", flush=True)
        try:
            for chunk in engine.generate_stream(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
            ):
                print(chunk, end="", flush=True)
        except KeyboardInterrupt:
            print(" [interrupted]", end="")
        print("\n")


if __name__ == "__main__":
    main()
