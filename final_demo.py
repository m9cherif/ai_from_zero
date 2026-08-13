"""Final demo: proper training with validation and generation."""
import os, sys, time, math
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

# Split data
with open("data/tinyshakespeare.txt", "r") as f:
    all_lines = [l.strip() for l in f if l.strip()]
split = int(len(all_lines) * 0.9)
train_lines = all_lines[:split]
val_lines = all_lines[split:]
print(f"Train: {len(train_lines)} lines, Val: {len(val_lines)} lines")

# Model
model = LanguageModel(LMConfig(
    vocab_size=tok.vocab_size, d_model=128, n_heads=4, d_ff=384,
    n_layers=5, max_seq_len=128, dropout=0.3,
    activation="silu", norm_type="rmsnorm", pre_norm=True,
    padding_idx=tok.pad_token_id,
)).to(device)
print(f"Model: {model.num_parameters():,} params")

opt = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.05)
sched = WarmupCosineLR(opt, warmup_steps=200, total_steps=8000, min_lr_ratio=0.05)
clip = GradientClipper(max_norm=1.0)
bb = BatchBuilder(pad_token_id=tok.pad_token_id, max_length=128, device=device)

def make_batch(lines, tokenizer, builder):
    line = random.choice(lines)
    ids = tokenizer.encode(line, add_special_tokens=True)
    if len(ids) < 10 or len(ids) > 128:
        return None
    batch = builder.build_batch([ids])
    return {k: v.to(device) for k, v in batch.items()}

import random
model.train()
step = 0
best_val = float("inf")
start = time.time()
patience = 0
MAX_STEPS = 5000

print(f"\n{'='*55}")
print("TRAINING")
print(f"{'='*55}")

while step < MAX_STEPS and patience < 10:
    batch = make_batch(train_lines, tok, bb)
    if batch is None:
        continue
    out = model(batch["input_ids"], labels=batch["labels"])
    loss = out["loss"]
    opt.zero_grad()
    loss.backward()
    clip.clip(model.parameters())
    opt.step()
    sched.step()
    step += 1

    if step % 200 == 0:
        # Validation step
        model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(min(50, len(val_lines))):
                vb = make_batch(val_lines, tok, bb)
                if vb is None:
                    continue
                vo = model(vb["input_ids"], labels=vb["labels"])
                val_losses.append(vo["loss"].item())
        avg_val = sum(val_losses) / len(val_losses) if val_losses else float("inf")
        ppl = math.exp(min(avg_val, 20))
        elapsed = time.time() - start
        lr = opt._param_groups[0]["lr"]
        print(f"  step {step:4d} | train {loss.item():.4f} | val {avg_val:.4f} | ppl {ppl:.2f} | lr {lr:.2e} | {elapsed:.0f}s")
        if avg_val < best_val:
            best_val = avg_val
            patience = 0
        else:
            patience += 1
        model.train()

print(f"\nTraining done in {time.time()-start:.0f}s (best val: {best_val:.4f})")

# Generation
model.eval()
print(f"\n{'='*55}")
print("GENERATION SAMPLES")
print(f"{'='*55}")

def gen(prompt, temp=0.5, top_k=8, rep=1.3):
    ids = tok.encode(prompt, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        g = model.generate(inp, max_new_tokens=80, temperature=temp,
            top_k=top_k, top_p=0.8, eos_token_id=tok.eos_token_id,
            repetition_penalty=rep)
    return tok.decode(g[0].tolist(), skip_special_tokens=True).strip()

prompts = [
    "To be, or not to be",
    "The king",
    "Now is the winter",
    "Friends, Romans",
    "But soft",
    "Romeo, Romeo",
]

for p in prompts:
    for t in [0.3, 0.5, 0.7]:
        text = gen(p, temp=t, top_k=6, rep=1.4)
        if len(text) > len(p) + 10:
            extra = text[len(p):].strip()
            print(f"\nPrompt: {p}")
            print(f"Model:  {p} {extra[:120]}")
            print(f"[temp={t}, +{len(extra)} chars]")
            break
    else:
        text = gen(p, temp=0.4, top_k=4, rep=1.5)
        extra = text[len(p):].strip()
        print(f"\nPrompt: {p}")
        print(f"Model:  {p} {extra[:120]}")
        print(f"[fallback]")
