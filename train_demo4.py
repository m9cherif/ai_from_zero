"""Training with improved generation."""
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
VOCAB = tok.vocab_size

print(f"Tokenizer: {VOCAB} tokens")

model = LanguageModel(LMConfig(
    vocab_size=VOCAB, d_model=96, n_heads=4, d_ff=256,
    n_layers=4, max_seq_len=256, dropout=0.1,
    activation="silu", norm_type="rmsnorm", pre_norm=True,
    padding_idx=tok.pad_token_id,
)).to(device)
print(f"Model: {model.num_parameters():,} params")

opt = AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
sched = WarmupCosineLR(opt, warmup_steps=50, total_steps=500, min_lr_ratio=0.1)
clip = GradientClipper(max_norm=1.0)
bb = BatchBuilder(pad_token_id=tok.pad_token_id, max_length=256, device=device)

with open("data/tinyshakespeare.txt", "r") as f:
    lines = [l.strip() for l in f if l.strip()]
print(f"Data: {len(lines)} lines")

MAX_STEPS = 500
step = 0
start = time.time()
import random

print(f"\n{'='*55}")
print("TRAINING")
print(f"{'='*55}")

model.train()
while step < MAX_STEPS:
    random.shuffle(lines)
    for line in lines:
        if step >= MAX_STEPS:
            break
        ids = tok.encode(line, add_special_tokens=True)
        if len(ids) < 10 or len(ids) > 256:
            continue
        batch = bb.build_batch([ids])
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"], labels=batch["labels"])
        loss = out["loss"]
        opt.zero_grad()
        loss.backward()
        clip.clip(model.parameters())
        opt.step()
        sched.step()
        step += 1
        if step % 200 == 0:
            elapsed = time.time() - start
            lr = opt._param_groups[0]["lr"]
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
    "Now is the winter",
]

import re

def generate_text(prompt, max_new=100, temp=0.8, top_k=20, top_p=0.9):
    """Generate without adding EOS to prompt."""
    # Encode prompt WITHOUT special tokens for the input
    ids = tok.encode(prompt, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        gen = model.generate(
            inp, max_new_tokens=max_new, temperature=temp,
            top_k=top_k, top_p=top_p,
            eos_token_id=tok.eos_token_id,
        )
    text = tok.decode(gen[0].tolist(), skip_special_tokens=True)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

for prompt in prompts:
    text = generate_text(prompt, max_new=80)
    print(f"\nPrompt: {prompt}")
    print(f"Gen:    {text}")
