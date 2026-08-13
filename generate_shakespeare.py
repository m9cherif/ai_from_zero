"""Generate Shakespeare text with the trained model."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from char_tokenizer import CharTokenizer
from myai.nn.model import LanguageModel, LMConfig

device = torch.device("cpu")
tok = CharTokenizer("output/tokenizer.json")

# Recreate the exact same model architecture
model = LanguageModel(LMConfig(
    vocab_size=tok.vocab_size, d_model=96, n_heads=4, d_ff=256,
    n_layers=4, max_seq_len=256, dropout=0.1,
    activation="silu", norm_type="rmsnorm", pre_norm=True,
    padding_idx=tok.pad_token_id,
)).to(device)

# Re-train quickly (1000 steps to get a model that's not fully overfit)
from myai.train.optimizer import AdamW
from myai.train.scheduler import WarmupCosineLR
from myai.train.gradient import GradientClipper
from myai.data.batching import BatchBuilder

opt = AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
sched = WarmupCosineLR(opt, warmup_steps=50, total_steps=1000, min_lr_ratio=0.1)
clip = GradientClipper(max_norm=1.0)
bb = BatchBuilder(pad_token_id=tok.pad_token_id, max_length=256, device=device)

with open("data/tinyshakespeare.txt", "r") as f:
    lines = [l.strip() for l in f if l.strip()]

import random
model.train()
for step in range(800):
    line = random.choice(lines)
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

print(f"Trained {step+1} steps, final loss: {loss.item():.4f}")
print()

model.eval()

def generate(prompt, max_new=150, temp=0.7, top_k=15, top_p=0.85, rep_penalty=1.2):
    ids = tok.encode(prompt, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        gen = model.generate(
            inp, max_new_tokens=max_new, temperature=temp,
            top_k=top_k, top_p=top_p,
            eos_token_id=tok.eos_token_id,
            repetition_penalty=rep_penalty,
        )
    text = tok.decode(gen[0].tolist(), skip_special_tokens=True)
    return text.strip()

# Try many different prompts
prompts = [
    "To be, or not to be",
    "Now is the winter",
    "Friends, Romans",
    "What light through yonder",
    "Romeo, Romeo",
    "But soft, what light",
]

for prompt in prompts:
    for temp in [0.5, 0.3]:
        text = generate(prompt, max_new=100, temp=temp, top_k=10, top_p=0.85, rep_penalty=1.3)
        if len(text) > len(prompt) + 5:
            print(f"\n[T={temp}] {prompt}")
            print(f"  {text}")
            break
    else:
        # If neither temperature worked, show the longer one
        text = generate(prompt, max_new=100, temp=0.4, top_k=8, top_p=0.8, rep_penalty=1.5)
        print(f"\n[fallback] {prompt}")
        print(f"  {text}")
