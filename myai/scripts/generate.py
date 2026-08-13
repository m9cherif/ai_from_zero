"""Generation script: generates text using a trained model."""

import argparse
import sys
from ..core.logging import logger
from ..inference.engine import InferenceEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text using a trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer file")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt")
    parser.add_argument("--prompt-file", type=str, default=None, help="File containing prompts (one per line)")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--min-p", type=float, default=None, help="Min-p sampling parameter")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Penalty for repeated tokens")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--stream", action="store_true", help="Print tokens as they are generated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Initialize inference engine
    device = None if args.device == "auto" else args.device
    engine = InferenceEngine(device=device)
    engine.load_checkpoint(args.checkpoint, args.tokenizer)

    def run(prompt: str) -> str:
        """Generate, streaming to stdout when requested."""
        kwargs = dict(
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.repetition_penalty,
        )
        if not args.stream:
            return engine.generate(prompt=prompt, **kwargs)

        print(prompt, end="", flush=True)
        pieces = []
        for chunk in engine.generate_stream(prompt=prompt, **kwargs):
            print(chunk, end="", flush=True)
            pieces.append(chunk)
        print()
        return prompt + "".join(pieces)

    if args.interactive:
        print("Interactive generation mode. Type 'quit' to exit.")
        print("=" * 50)
        while True:
            try:
                prompt = input("\nPrompt: ")
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                if not prompt.strip():
                    continue

                if not args.stream:
                    print("\nGenerated:")
                generated = run(prompt)
                if not args.stream:
                    print(generated)
                print("=" * 50)
            except KeyboardInterrupt:
                print("\nExiting...")
                break
    elif args.prompt:
        generated = run(args.prompt)
        if not args.stream:
            print(generated)
    elif args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                prompt = line.strip()
                if prompt:
                    print(f"Prompt: {prompt}")
                    generated = run(prompt)
                    if not args.stream:
                        print(f"Generated: {generated}")
                    print("-" * 40)
    else:
        logger.error("Provide a prompt, prompt file, or use --interactive mode")
        sys.exit(1)


if __name__ == "__main__":
    main()
