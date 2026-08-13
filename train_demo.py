"""Complete end-to-end training script for the language model.

Trains on TinyShakespeare, demonstrating the entire pipeline:
1. Tokenizer training (already done)
2. Model configuration
3. Training loop
4. Text generation
"""

import os
import time
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from myai.config.presets import TrainConfig
from myai.core.random import set_seed
from myai.core.logging import logger, LogLevel
from myai.tokenizer import BPETokenizer
from myai.nn.model import LanguageModel, LMConfig
from myai.train.optimizer import AdamW
from myai.train.scheduler import WarmupCosineLR
from myai.train.gradient import GradientClipper
from myai.train.loop import TrainingLoop
from myai.data.streaming import StreamingDataset
from myai.data.batching import BatchBuilder
from myai.inference import AutoregressiveGenerator

logger.set_level(LogLevel.INFO)

# ============================================================
# Configuration
# ============================================================
DEVICE = torch.device("cpu")
VOCAB_SIZE = 256
MAX_SEQ_LEN = 64
BATCH_SIZE = 8
EMBED_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
NUM_LAYERS = 3
MAX_STEPS = 500
LEARNING_RATE = 1e-3
WARMUP_STEPS = 100
EVAL_EVERY = 100
LOG_EVERY = 10

print("=" * 60)
print("MYAI LANGUAGE MODEL TRAINING DEMO")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Model: {NUM_LAYERS} layers, {EMBED_DIM} dim, {NUM_HEADS} heads")
print(f"Max steps: {MAX_STEPS}, Batch size: {BATCH_SIZE}")
print(f"Max seq len: {MAX_SEQ_LEN}, Vocab: {VOCAB_SIZE}")
print("=" * 60)

# ============================================================
# Load tokenizer
# ============================================================
print("\n[1/4] Loading tokenizer...")
tokenizer = BPETokenizer.load("output/tokenizer.json")
print(f"  Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

# ============================================================
# Create model
# ============================================================
print("\n[2/4] Creating model...")
model_config = LMConfig(
    vocab_size=VOCAB_SIZE,
    d_model=EMBED_DIM,
    n_heads=NUM_HEADS,
    d_ff=FF_DIM,
    n_layers=NUM_LAYERS,
    max_seq_len=MAX_SEQ_LEN,
    dropout=0.1,
    activation="silu",
    norm_type="rmsnorm",
    pre_norm=True,
    padding_idx=tokenizer.pad_token_id,
)

model = LanguageModel(model_config).to(DEVICE)
total_params = model.num_parameters()
print(f"  Model created: {total_params:,} parameters")

# ============================================================
# Create optimizer and scheduler
# ============================================================
optimizer = AdamW(
    parameters=model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1,
)

scheduler = WarmupCosineLR(
    optimizer=optimizer,
    warmup_steps=WARMUP_STEPS,
    total_steps=MAX_STEPS,
    min_lr_ratio=0.1,
)

gradient_clipper = GradientClipper(max_norm=1.0)

# ============================================================
# Create data pipeline
# ============================================================
print("\n[3/4] Creating data pipeline...")

class TextFileDataset:
    """Simple dataset that reads lines from a text file."""
    def __init__(self, path, tokenizer, max_seq_len):
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        with open(path, "r", encoding="utf-8") as f:
            self._lines = [l.strip() for l in f if l.strip()]
        print(f"  Loaded {len(self._lines)} lines from {path}")

    def __iter__(self):
        import random
        indices = list(range(len(self._lines)))
        random.shuffle(indices)
        for i in indices:
            text = self._lines[i]
            ids = self._tokenizer.encode(text, add_special_tokens=True)
            if len(ids) > self._max_seq_len:
                ids = ids[:self._max_seq_len]
            if len(ids) >= 4:  # Need at least some tokens
                yield ids

train_data = TextFileDataset("data/tinyshakespeare.txt", tokenizer, MAX_SEQ_LEN)

# ============================================================
# Training loop
# ============================================================
print("\n[4/4] Starting training...")
print("=" * 60)

model.train()
global_step = 0
running_loss = 0.0
running_tokens = 0
start_time = time.time()

batch_builder = BatchBuilder(
    pad_token_id=tokenizer.pad_token_id,
    max_length=MAX_SEQ_LEN,
    device=DEVICE,
)

for epoch in range(5):
    for token_ids in train_data:
        if global_step >= MAX_STEPS:
            break

        # Build batch
        batch = batch_builder.build_batch([token_ids])

        # Forward pass
        outputs = model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            attention_mask=batch.get("attention_mask"),
        )
        loss = outputs["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = gradient_clipper.clip(model.parameters())

        # Optimizer step
        optimizer.step()
        scheduler.step()

        # Track metrics
        step_loss = loss.item()
        tokens = (batch["labels"] != -100).sum().item()
        running_loss += step_loss
        running_tokens += tokens
        global_step += 1

        # Logging
        if global_step % LOG_EVERY == 0:
            elapsed = time.time() - start_time
            avg_loss = running_loss / max(LOG_EVERY, 1)
            tps = running_tokens / max(elapsed, 0.001)
            current_lr = optimizer._param_groups[0]["lr"]
            print(
                f"  Step {global_step:4d}/{MAX_STEPS} | "
                f"loss: {avg_loss:.4f} | "
                f"lr: {current_lr:.2e} | "
                f"grad: {grad_norm:.2f} | "
                f"tok/s: {tps:.0f} | "
                f"elapsed: {elapsed:.0f}s"
            )
            running_loss = 0.0
            running_tokens = 0

        # Evaluation
        if global_step % EVAL_EVERY == 0:
            model.eval()
            # Generate sample text
            prompt = "To be, or not to be"
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

            with torch.no_grad():
                generated = model.generate(
                    input_tensor,
                    max_new_tokens=30,
                    temperature=0.8,
                    top_k=20,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id,
                )
            generated_text = tokenizer.decode(
                generated[0].tolist(),
                skip_special_tokens=True,
            )
            print(f"\n  === Generation at step {global_step} ===")
            print(f"  {generated_text}")
            print(f"  ====================================\n")
            model.train()

    if global_step >= MAX_STEPS:
        break

total_time = time.time() - start_time
print("=" * 60)
print(f"TRAINING COMPLETE in {total_time:.1f}s ({global_step} steps)")
print("=" * 60)

# ============================================================
# Final generation demo
# ============================================================
print("\n" + "=" * 60)
print("FINAL TEXT GENERATION DEMO")
print("=" * 60)

model.eval()
prompts = [
    "To be, or not to be",
    "The king",
    "Romeo, Romeo",
    "Friends, Romans, countrymen",
    "What light through yonder window breaks",
]

for prompt in prompts:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        generated = model.generate(
            input_tensor,
            max_new_tokens=40,
            temperature=0.9,
            top_k=30,
            top_p=0.92,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {full_text}")
    print("-" * 50)
