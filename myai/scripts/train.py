"""Training script: trains a language model from scratch."""

import argparse
import sys
import os
from pathlib import Path

from ..config.presets import TrainConfig
from ..core.logging import logger, LogLevel
from ..core.random import set_seed
from ..train.engine import Trainer
from ..data.streaming import StreamingDataset
from ..tokenizer import load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a language model from scratch")
    parser.add_argument("--config", type=str, default=None, help="Path to training configuration file")
    parser.add_argument("--data", type=str, nargs="+", default=None, help="Data file paths")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer file")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto/cpu/cuda)")
    parser.add_argument("--log-level", type=str, default="info", help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logging
    log_level_map = {
        "trace": LogLevel.TRACE,
        "debug": LogLevel.DEBUG,
        "info": LogLevel.INFO,
        "warning": LogLevel.WARNING,
        "error": LogLevel.ERROR,
        "fatal": LogLevel.FATAL,
    }
    logger.set_level(log_level_map.get(args.log_level.lower(), LogLevel.INFO))

    # Load configuration
    if args.config:
        config = TrainConfig.load(args.config)
    else:
        config = TrainConfig()

    # Override with command-line arguments
    if args.seed:
        config.seed = args.seed
    if args.device:
        config.hardware.device = args.device
    if args.data:
        config.data.data_paths = args.data
    if args.tokenizer:
        config.tokenizer.vocab_size = None  # Will use tokenizer's vocab
    if args.output_dir:
        config.checkpoint.save_dir = os.path.join(args.output_dir, "checkpoints")
        config.logging.log_file = os.path.join(args.output_dir, "training.log")
    if args.resume:
        config.checkpoint.resume_from = args.resume

    # Set random seed
    set_seed(config.seed)

    # Load tokenizer (BPE or character-level, detected from the file)
    tokenizer_path = args.tokenizer
    if tokenizer_path and os.path.exists(tokenizer_path):
        tokenizer = load_tokenizer(tokenizer_path)
        config.model.vocab_size = tokenizer.vocab_size
        logger.info(
            f"Loaded {type(tokenizer).__name__} with vocab_size={tokenizer.vocab_size}"
        )
    else:
        logger.error(f"Tokenizer not found at {tokenizer_path}. Train a tokenizer first.")
        sys.exit(1)

    # Create dataset
    dataset = StreamingDataset(
        data_paths=config.data.data_paths,
        tokenizer=tokenizer,
        max_seq_len=config.data.max_seq_len,
        shuffle_buffer_size=config.data.shuffle_buffer_size,
        pack_sequences=config.data.pack_sequences,
    )
    dataset.discover_files()

    # Passing the tokenizer lets the trainer embed it in every checkpoint, so a
    # checkpoint is self-contained for inference.
    trainer = Trainer(config, tokenizer=tokenizer)
    trainer.train(dataset)

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
