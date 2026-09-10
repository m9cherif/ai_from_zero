"""Build a tokenizer from the corpus in data/.

    python scripts/build_tokenizer.py --type char
    python scripts/build_tokenizer.py --type bpe --vocab-size 4096
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myai.tokenizer import CharTokenizer, TokenizerTrainer, load_tokenizer


def find_corpus(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern)))
    return [p for p in paths if os.path.isfile(p)]


def read_texts(paths, max_chars: int = 0):
    """Yield file contents, optionally stopping after max_chars in total.

    A vocabulary is fitted on a sample, not the whole corpus: BPE training cost
    grows with the text it sees, and merges learned from 20 MB are effectively
    the same as merges learned from 20 GB. Reading everything would turn a
    two-minute step into hours.
    """
    budget = max_chars or float("inf")
    per_file = (max_chars // max(len(paths), 1)) if max_chars else None
    for path in paths:
        if budget <= 0:
            return
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read(per_file) if per_file else f.read()
        budget -= len(text)
        yield text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", default=["data/*.txt", "data/books/*.txt"])
    parser.add_argument("--output", default="output/tokenizer.json")
    parser.add_argument("--type", choices=["char", "bpe"], default="char")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument(
        "--max-chars", type=int, default=200_000_000,
        help="Characters of corpus to fit the vocabulary on, spread across "
             "files. 0 uses everything, which on a multi-gigabyte corpus takes "
             "hours for no gain.",
    )
    args = parser.parse_args()

    paths = find_corpus(args.data)
    if not paths:
        print(f"No corpus files matched {args.data}", file=sys.stderr)
        sys.exit(1)

    total_mb = sum(os.path.getsize(p) for p in paths) / 1e6
    print(f"Corpus: {len(paths)} files, {total_mb:.1f} MB")

    if args.type == "char":
        tokenizer = CharTokenizer.train(read_texts(paths, args.max_chars),
                                        min_frequency=args.min_frequency)
    else:
        trainer = TokenizerTrainer(
            target_vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
        )
        tokenizer = trainer.train(read_texts(paths, args.max_chars))

    tokenizer.save(args.output)

    # Prove the round trip is lossless before anything trains on it.
    sample = "To be, or not to be:\nthat is the question."
    restored = tokenizer.decode(tokenizer.encode(sample, add_special_tokens=False))
    ids = tokenizer.encode(sample, add_special_tokens=False)

    print(f"\nSaved {args.type} tokenizer to {args.output}")
    print(f"  vocab size:      {tokenizer.vocab_size}")
    print(f"  compression:     {len(sample) / max(len(ids), 1):.2f} chars/token")
    print(f"  lossless decode: {restored == sample}")
    if restored != sample:
        print(f"    expected {sample!r}\n    got      {restored!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
