"""Evaluation script: evaluates a trained model on benchmarks."""

import argparse
import json
import sys
from ..core.logging import logger
from ..inference.engine import InferenceEngine
from ..evaluate.benchmark import Benchmark
from ..data.streaming import StreamingDataset
from ..tokenizer import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained language model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer file")
    parser.add_argument("--eval-data", type=str, default=None, help="Evaluation dataset")
    parser.add_argument("--prompts", type=str, default=None, help="File with prompts for generation evaluation")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Output file for results")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load model
    engine = InferenceEngine(device=args.device)
    engine.load_checkpoint(args.checkpoint, args.tokenizer)

    # Load tokenizer (BPE or character-level, detected from the file)
    tokenizer = load_tokenizer(args.tokenizer)

    # Create benchmark
    benchmark = Benchmark(model=engine.model, tokenizer=tokenizer)

    # Load evaluation data
    eval_dataset = None
    if args.eval_data:
        eval_dataset = StreamingDataset(
            data_paths=[args.eval_data],
            tokenizer=tokenizer,
            max_seq_len=engine.model.config.max_seq_len,
        )
        eval_dataset.discover_files()

    # Load prompts
    prompts = None
    if args.prompts:
        with open(args.prompts, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]

    # Run evaluation
    results = benchmark.full_evaluation(
        eval_dataset=eval_dataset,
        prompts=prompts,
    )

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Evaluation results saved to {args.output}")


if __name__ == "__main__":
    main()
