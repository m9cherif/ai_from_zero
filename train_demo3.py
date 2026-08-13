"""Training with working character-level tokenizer."""
import os, sys, time
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

tok = CharTokenizer("output/tokenizer.json")
print(f"Tokenizer: {tok.vocab_size} tokens, space_id={tok.vocab.get(' ')}")

model = LanguageModel(LMConfig(
    vocab_size=tok.vocab_size,
    d_model=96,
    n_heads=4,
    d_ff=256,
    n_layers=4,
    max_seq_len=128,
    dropout=0.1,
    activation="silu",
    norm_type="rmsnorm",
    pre_norm=True,
    padding_idx=tok.pad_token_id,
)).to(device)

print(f"Model: {model.num_parameters():,} params")

optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
scheduler = WarmupCosineLR(optimizer, warmup_steps=50, total_steps=2000, min_lr_ratio=0.1)
clipper = GradientClipper(max_norm=1.0)
bb = BatchBuilder(pad_token_id=tok.pad_token_id, max_length=128, device=device)

with open("data/tinyshakespeare.txt", "r") as f:
    raw_lines = [l.strip() for l in f if l.strip()]
print(f"Data: {len(raw_lines)} lines")

MAX_STEPS = 2000
step = 0
start = time.time()
import random

print(f"\n{'='*55}")
print("TRAINING")
print(f"{'='*55}")

model.train()
while step < MAX_STEPS:
    random.shuffle(raw_lines)
    for line in raw_lines:
        if step >= MAX_STEPS:
            break
        ids = tok.encode(line, add_special_tokens=True)
        if len(ids) < 8 or len(ids) > 128:
            continue
        batch = bb.build_batch([ids])
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"], labels=batch["labels"])
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        clipper.clip(model.parameters())
        optimizer.step()
        scheduler.step()
        step += 1
        if step % 100 == 0:
            elapsed = time.time() - start
            lr = optimizer._param_groups[0]["lr"]
            print(f"  step {step:4d}/{MAX_STEPS} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s")

print(f"\n{'='*55}")
print(f"DONE in {time.time()-start:.0f}s")
print(f"{'='*55}")

print(f"\n{'='*55}")
print("GENERATION")
print(f"{'='*55}")

model.eval()
prompts = [
    "To be, or not to be",
    "The king",
    "Romeo, Romeo",
    "Friends, Romans, countrymen",
    "What light through yonder window breaks",
    "Now is the winter of our discontent",
]

for prompt in prompts:
    ids = tok.encode(prompt, add_special_tokens=True)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        gen = model.generate(inp, max_new_tokens=60, temperature=0.9, top_k=30, top_p=0.92, eos_token_id=tok.eos_token_id)
    text = tok.decode(gen[0].tolist(), skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Gen:    {text}")
