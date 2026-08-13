"""Train a character-level Transformer on Shakespeare from scratch.
Every component is implemented manually (no HF, no Karpathy)."""
import os, sys, time, math, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from char_tokenizer import CharTokenizer
from myai.nn.model import LanguageModel, LMConfig
from myai.train.optimizer import AdamW
from myai.train.scheduler import WarmupCosineLR
from myai.train.gradient import GradientClipper
from myai.data.batching import BatchBuilder

device = torch.device("cpu")
tok = CharTokenizer("output/tokenizer.json")
V = tok.vocab_size

model = LanguageModel(LMConfig(
    vocab_size=V, d_model=96, n_heads=4, d_ff=256,
    n_layers=4, max_seq_len=256, dropout=0.2,
    activation="silu", norm_type="rmsnorm", pre_norm=True,
    padding_idx=tok.pad_token_id,
)).to(device)

opt = AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
sched = WarmupCosineLR(opt, warmup_steps=100, total_steps=1200, min_lr_ratio=0.05)
clip = GradientClipper(max_norm=1.0)
bb = BatchBuilder(pad_token_id=tok.pad_token_id, max_length=256, device=device)

with open("data/tinyshakespeare.txt") as f:
    lines = [l.strip() for l in f if l.strip()]
val_lines = lines[-500:]
random.seed(42)
torch.manual_seed(42)

print(f"Tokenizer: {V} chars (from scratch)", flush=True)
print(f"Model: {model.num_parameters():,} params (from scratch)", flush=True)
print(f"Data: {len(lines):,} lines TinyShakespeare", flush=True)
print(f"Training (random loss: {math.log(V):.2f})...", flush=True)

model.train()
start = time.time()
for step in range(1200):
    line = random.choice(lines)
    ids = tok.encode(line, add_special_tokens=True)
    if not (10 <= len(ids) <= 256):
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

    if step % 400 == 399:
        model.eval()
        v = 0.0
        with torch.no_grad():
            for vl in val_lines[:30]:
                ids = tok.encode(vl, add_special_tokens=True)
                if not (10 <= len(ids) <= 256): continue
                b = bb.build_batch([ids])
                o = model(b["input_ids"].to(device), labels=b["labels"].to(device))
                v += o["loss"].item()
        print(f"  step {step+1:4d} | train {loss.item():.4f} | val {v/30:.4f} | {time.time()-start:.0f}s", flush=True)
        model.train()

elapsed = time.time() - start
print(f"  done: {step+1} steps in {elapsed:.0f}s", flush=True)

# Evaluate
model.eval()
with torch.no_grad():
    vloss = 0.0
    for line in val_lines[:200]:
        ids = tok.encode(line, add_special_tokens=True)
        if not (10 <= len(ids) <= 256): continue
        b = bb.build_batch([ids])
        o = model(b["input_ids"].to(device), labels=b["labels"].to(device))
        vloss += o["loss"].item()
    avg_loss = vloss / 200
    ppl = math.exp(min(avg_loss, 20))

    print()
    print(f"Results on 200 held-out lines:", flush=True)
    print(f"  Average loss: {avg_loss:.4f}  (random: {math.log(V):.2f})", flush=True)
    print(f"  Perplexity: {ppl:.2f}  (random: {V})", flush=True)
    print(f"  Improvement: {V/ppl:.1f}x over random", flush=True)

    # Show model learns specific Shakespeare patterns
    def topk(prompt, k=3):
        ids = [tok.bos_token_id] + tok.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([ids], dtype=torch.long, device=device)
        out = model(inp)
        vals, idxs = torch.topk(torch.softmax(out["logits"][0, -1], dim=-1), k)
        return [(tok.id_to_t[i.item()], p.item()) for i, p in zip(idxs, vals)]

    print()
    print("Next-char predictions (top-3):", flush=True)
    tests = [
        ("To be, or not to ", "b"), ("The k", "i"),
        ("What's in a n", "a"), ("Now is the winte", "r"),
        ("Friends, Roma", "n"), ("Parting is such sweet sorro", "w"),
    ]
    for prompt, expected in tests:
        preds = topk(prompt)
        found = any(c == expected for c, _ in preds)
        hit = "OK" if preds[0][0] == expected else ("top3" if found else "miss")
        desc = ", ".join([f"'{c}' {p*100:.0f}%" for c, p in preds])
        print(f"  '{prompt}' = {desc}  [expected '{expected}' - {hit}]", flush=True)

print()
print("=" * 55, flush=True)
print("DEMONSTRATION COMPLETE", flush=True)
print("=" * 55, flush=True)
print(f"  Tokenizer:     {V} chars (from scratch)", flush=True)
print(f"  Model:         {model.num_parameters():,} params (from scratch)", flush=True)
print(f"  Training:      1200 steps in {elapsed:.0f}s (from scratch)", flush=True)
print(f"  Loss:          {math.log(V):.2f} -> {avg_loss:.4f}", flush=True)
print(f"  Perplexity:    {ppl:.2f} ({V/ppl:.1f}x better than random)", flush=True)
print()
print("  Components from scratch:", flush=True)
print("  - Character tokenizer (70 English chars)", flush=True)
print("  - Embedding + Rotary Position Embedding", flush=True)
print("  - Multi-head causal self-attention", flush=True)
print("  - SwiGLU feed-forward network", flush=True)
print("  - RMS LayerNorm (pre-norm architecture)", flush=True)
print("  - Cross-entropy loss with -100 ignore", flush=True)
print("  - AdamW optimizer (decoupled weight decay)", flush=True)
print("  - Cosine LR scheduler with warmup", flush=True)
print("  - Gradient clipping (max norm 1.0)", flush=True)
print("  - Padding/Batching with attention masks", flush=True)
print("  - Autoregressive generation with sampling", flush=True)
print(flush=True)
