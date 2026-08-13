"""Complete training demo with character-level tokenizer."""

import os, sys, time, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from char_tokenizer import CharTokenizer
from myai.core.logging import logger, LogLevel
from myai.nn.model import LanguageModel, LMConfig
from myai.train.optimizer import AdamW
from myai.train.scheduler import WarmupCosineLR
from myai.train.gradient import GradientClipper
from myai.data.batching import BatchBuilder

logger.set_level(LogLevel.WARNING)
device = torch.device("cpu")

tokenizer = CharTokenizer("output/tokenizer.json")
VOCAB_SIZE = tokenizer.vocab_size
print(f"Tokenizer: {VOCAB_SIZE} tokens")

# Tiny model for CPU
model = LanguageModel(LMConfig(
    vocab_size=VOCAB_SIZE,
    d_model=96,
    n_heads=4,
    d_ff=256,
    n_layers=4,
    max_seq_len=128,
    dropout=0.1,
    activation="silu",
    norm_type="rmsnorm",
    pre_norm=True,
    padding_idx=tokenizer.pad_token_id,
)).to(device)

print(f"Model: {model.num_parameters():,} params")

optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
scheduler = WarmupCosineLR(optimizer, warmup_steps=50, total_steps=2000, min_lr_ratio=0.1)
clipper = GradientClipper(max_norm=1.0)
batch_builder = BatchBuilder(pad_token_id=tokenizer.pad_token_id, max_length=128, device=device)

# Load data
with open("data/tinyshakespeare.txt", "r") as f:
    lines = [l.strip() for l in f if l.strip()]
print(f"Data: {len(lines)} lines")

MAX_STEPS = 1000
step = 0
start = time.time()

print(f"\n{'='*60}")
print(f"Training ...")
print(f"{'='*60}")

model.train()
while step < MAX_STEPS:
    import random
    random.shuffle(lines)
    for line in lines:
        if step >= MAX_STEPS:
            break
        ids = tokenizer.encode(line, add_special_tokens=True)
        if len(ids) < 8 or len(ids) > 128:
            continue
        batch = batch_builder.build_batch([ids])
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"], labels=batch["labels"])
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        clipper.clip(model.parameters())
        optimizer.step()
        scheduler.step()
        step += 1
        if step % 50 == 0:
            elapsed = time.time() - start
            lr_now = optimizer._param_groups[0]["lr"]
            print(f"  step {step:4d}/{MAX_STEPS} | loss {loss.item():.4f} | lr {lr_now:.2e} | {elapsed:.0f}s")

print(f"\n{'='*60}")
print(f"Training done in {time.time()-start:.0f}s")
print(f"{'='*60}")

# Generate
print(f"\n{'='*60}")
print(f"Generation")
print(f"{'='*60}")

model.eval()
prompts = [
    "To be, or not to be",
    "The king",
    "Romeo, Romeo",
    "Friends, Romans, countrymen",
]

for prompt in prompts:
    ids = tokenizer.encode(prompt, add_special_tokens=True)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        gen = model.generate(inp, max_new_tokens=50, temperature=0.9, top_k=30, top_p=0.92, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(gen[0].tolist(), skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Gen:    {text}")
