"""Quick demo: train for fewer steps, show short completions work."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from char_tokenizer import CharTokenizer
from myai.nn.model import LanguageModel, LMConfig
from myai.train.optimizer import AdamW
from myai.train.scheduler import WarmupCosineLR
from myai.train.gradient import GradientClipper
from myai.data.batching import BatchBuilder

device = torch.device("cpu")
tok = CharTokenizer("output/tokenizer.json")

with open("data/tinyshakespeare.txt", "r") as f:
    lines = [l.strip() for l in f if l.strip()]
print(f"Data: {len(lines)} lines")

model = LanguageModel(LMConfig(
    vocab_size=tok.vocab_size, d_model=128, n_heads=4, d_ff=384,
    n_layers=5, max_seq_len=128, dropout=0.2,
    activation="silu", norm_type="rmsnorm", pre_norm=True,
    padding_idx=tok.pad_token_id,
)).to(device)
print(f"Model: {model.num_parameters():,} params")

opt = AdamW(model.parameters(), lr=8e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.05)
sched = WarmupCosineLR(opt, warmup_steps=50, total_steps=1500, min_lr_ratio=0.1)
clip = GradientClipper(max_norm=1.0)
bb = BatchBuilder(pad_token_id=tok.pad_token_id, max_length=128, device=device)

import random
model.train()
step = 0
start = time.time()
N = 600

print(f"\nTraining {N} steps...")
while step < N:
    line = random.choice(lines)
    ids = tok.encode(line, add_special_tokens=True)
    if len(ids) < 10 or len(ids) > 128:
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
    if step % 100 == 0:
        print(f"  step {step}: loss {loss.item():.4f} | {time.time()-start:.0f}s")

print(f"\nTrained {step} steps in {time.time()-start:.0f}s")
print(f"\n{'='*55}")
print("GENERATION")
print(f"{'='*55}")

model.eval()

def gen(prompt, max_new=40, temp=0.4, top_k=6, rep=1.3):
    ids = tok.encode(prompt, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        g = model.generate(inp, max_new_tokens=max_new, temperature=temp,
            top_k=top_k, top_p=0.85, eos_token_id=tok.eos_token_id,
            repetition_penalty=rep)
    return tok.decode(g[0].tolist(), skip_special_tokens=True)

prompts = [
    "To be, or not to be",
    "The king",
    "Now is the winter",
    "Friends, Romans",
    "But soft, what light",
    "Romeo, Romeo",
    "All that glitters is",
    "A horse, a horse",
    "Parting is such",
    "What's in a name",
]

print()
for p in prompts:
    text = gen(p, max_new=30, temp=0.35, top_k=5, rep=1.4)
    extra = text[len(p):].strip()
    if extra:
        print(f"  {p} {extra[:60]}")
    else:
        print(f"  {p} [no continuation]")
