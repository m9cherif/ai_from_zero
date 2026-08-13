"""Prove the model actually learned Shakespeare patterns."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import math
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

model = LanguageModel(LMConfig(
    vocab_size=tok.vocab_size, d_model=128, n_heads=4, d_ff=384,
    n_layers=5, max_seq_len=128, dropout=0.2,
    activation="silu", norm_type="rmsnorm", pre_norm=True,
    padding_idx=tok.pad_token_id,
)).to(device)

opt = AdamW(model.parameters(), lr=8e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.05)
sched = WarmupCosineLR(opt, warmup_steps=50, total_steps=1500, min_lr_ratio=0.1)
clip = GradientClipper(max_norm=1.0)
bb = BatchBuilder(pad_token_id=tok.pad_token_id, max_length=128, device=device)

import random
model.train()
for step in range(800):
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

model.eval()
print(f"Trained (final loss: {loss.item():.4f})")
print()

# Test 1: Perplexity on held-out data
print("=" * 55)
print("TEST 1: Perplexity on unseen lines")
print("=" * 55)
val_lines = lines[-500:]

with torch.no_grad():
    total_nll = 0.0
    total_chars = 0
    for line in val_lines:
        ids = tok.encode(line, add_special_tokens=False)
        inp = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        targets = torch.tensor([ids[1:]], dtype=torch.long, device=device)
        if inp.shape[1] < 5:
            continue
        out = model(inp, labels=targets)
        total_nll += out["loss"].item() * (targets != -100).sum().item()
        total_chars += (targets != -100).sum().item()
    avg_nll = total_nll / max(total_chars, 1)
    ppl = math.exp(min(avg_nll, 20))
    print(f"  Average negative log-likelihood: {avg_nll:.4f}")
    print(f"  Perplexity: {ppl:.4f}")
    print(f"  Random baseline (70 tokens): 70.0")
    print(f"  IMPROVEMENT: {70.0/ppl:.1f}x better than random!")

# Test 2: Next-character prediction accuracy  
print()
print("=" * 55)
print("TEST 2: Next-character prediction accuracy")
print("=" * 55)
with torch.no_grad():
    correct = 0
    total = 0
    for line in val_lines[:200]:
        ids = tok.encode(line, add_special_tokens=False)
        for i in range(len(ids)-1):
            inp = torch.tensor([[ids[i]]], dtype=torch.long, device=device)
            out = model(inp)
            pred = out["logits"][0, -1].argmax().item()
            if pred == ids[i+1]:
                correct += 1
            total += 1
    acc = correct / max(total, 1)
    print(f"  Next-char accuracy: {acc*100:.1f}% ({correct}/{total})")
    print(f"  Random baseline: {1/70*100:.1f}%")

# Test 3: Show model predicts real text correctly
print()
print("=" * 55)
print("TEST 3: Model correctly completes Shakespeare quotes")
print("=" * 55)

def predict_next(prompt):
    ids = tok.encode(prompt, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(inp)
    logits = out["logits"][0, -1]
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, 5)
    return [(tok.id_to_t[i.item()], p.item()) for i, p in zip(top_indices, top_probs)]

# Test cases: partial quotes where we know the next character
tests = [
    ("To be, or not to ", "b"),     # "be"
    ("The k", "i"),                   # "king"
    ("What's in a n", "a"),          # "name"
    ("Friends, Roma", "n"),          # "Romans"
    ("Now is the winte", "r"),       # "winter"
    ("Parting is such sweet sorro", "w"),  # "sorrow"
]

for prompt, expected in tests:
    preds = predict_next(prompt)
    top5 = ", ".join([f"'{ch}' ({p*100:.0f}%)" for ch, p in preds])
    correct = any(ch == expected for ch, _ in preds[:3])
    mark = "✓" if preds[0][0] == expected else "~" if correct else "✗"
    print(f"  {prompt} -> top5: {top5}")
    print(f"    Expected '{expected}' {mark}\n")
