"""Tokenizer training script: trains a BPE tokenizer on text data."""

import argparse
import sys
from pathlib import Path
from ..tokenizer import train_tokenizer_from_files, CharTokenizer
from ..core.logging import logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tokenizer on text files")
    parser.add_argument("--input", type=str, nargs="+", required=True, help="Input text files or directories")
    parser.add_argument("--output", type=str, default="./tokenizer.json", help="Output path for the tokenizer")
    parser.add_argument("--type", type=str, default="bpe", choices=["bpe", "char"],
                        help="Tokenizer type: subword BPE or character-level")
    parser.add_argument("--vocab-size", type=int, default=8192, help="Target vocabulary size (BPE only)")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum pair frequency")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Collect file paths
    file_paths = []
    for path in args.input:
        p = Path(path)
        if p.is_file():
            file_paths.append(str(p.resolve()))
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in {".txt", ".md", ".json", ".jsonl", ".csv"}:
                    file_paths.append(str(f.resolve()))
        else:
            logger.warning(f"Path not found: {path}")

    if not file_paths:
        logger.error("No input files found")
        sys.exit(1)

    logger.info(
        f"Training {args.type} tokenizer on {len(file_paths)} files "
        f"(vocab_size={args.vocab_size})"
    )

    if args.type == "char":
        def texts():
            for path in file_paths:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    yield f.read()

        tokenizer = CharTokenizer.train(texts(), min_frequency=args.min_frequency)
        tokenizer.save(args.output)
    else:
        tokenizer = train_tokenizer_from_files(
            file_paths=file_paths,
            target_vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            save_path=args.output,
        )

    logger.info(f"Tokenizer saved to {args.output}")
    logger.info(f"Vocabulary size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
