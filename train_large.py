"""Train on 14M+ chars from classic literature (expanded dataset)."""
import os, sys, time, math, random, glob
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

# Load all book files + Shakespeare
import glob as g
paths = ["data/tinyshakespeare.txt"] + g.glob("data/books/*.txt")
all_text = []
total_chars = 0
for path in paths:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    # Skip Gutenberg headers/footers
    start = text.find("*** START OF")
    end = text.find("*** END OF")
    if start >= 0:
        text = text[start:]
    if end >= 0:
        text = text[:end]
    chars = len(text)
    total_chars += chars
    all_text.append(text)
    print(f"  {os.path.basename(path)}: {chars:,} chars", flush=True)

# Split into chunks of max_seq_len characters
SEQ_LEN = 256
chunks = []
for text in all_text:
    for i in range(0, len(text) - SEQ_LEN, SEQ_LEN // 2):  # 50% overlap
        chunk = text[i:i + SEQ_LEN]
        # Only keep chunks with mostly printable chars
        printable = sum(1 for c in chunk if c.isprintable() or c in "\n\r\t")
        if printable > SEQ_LEN * 0.8:
            chunks.append(chunk)

print(f"\nTotal: {total_chars:,} chars -> {len(chunks):,} training chunks", flush=True)
print(f"Model context: {SEQ_LEN} chars", flush=True)

model = LanguageModel(LMConfig(
    vocab_size=V, d_model=96, n_heads=4, d_ff=256,
    n_layers=4, max_seq_len=SEQ_LEN, dropout=0.3,
    activation="silu", norm_type="rmsnorm", pre_norm=True,
    padding_idx=tok.pad_token_id,
)).to(device)
print(f"Model: {model.num_parameters():,} params", flush=True)

opt = AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.05)
sched = WarmupCosineLR(opt, warmup_steps=200, total_steps=10000, min_lr_ratio=0.05)
clip = GradientClipper(max_norm=1.0)
bb = BatchBuilder(pad_token_id=tok.pad_token_id, max_length=SEQ_LEN, device=device)

random.seed(42)
torch.manual_seed(42)

# Training
BATCH_SIZE = 8
total_steps = 10000
model.train()
start = time.time()

for step in range(total_steps):
    # Sample a batch of chunks
    batch_chunks = random.choices(chunks, k=BATCH_SIZE)
    batch_ids = [tok.encode(c, add_special_tokens=True) for c in batch_chunks]
    # Truncate to SEQ_LEN
    batch_ids = [ids[:SEQ_LEN] for ids in batch_ids]
    batch = bb.build_batch(batch_ids)
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(batch["input_ids"], labels=batch["labels"])
    loss = out["loss"]
    opt.zero_grad()
    loss.backward()
    clip.clip(model.parameters())
    opt.step()
    sched.step()

    if step % 1000 == 0 and step > 0:
        elapsed = time.time() - start
        lr = opt._param_groups[0]["lr"]
        print(f"  step {step:5d}/{total_steps} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.0f}s", flush=True)

elapsed = time.time() - start
print(f"\nTrained {total_steps} steps in {elapsed:.0f}s", flush=True)
print(f"Loss: {math.log(V):.2f} (random) -> {loss.item():.4f} (trained)", flush=True)

# Evaluate
model.eval()
with torch.no_grad():
    vloss = 0.0; vn = 0
    for chunk in random.sample(chunks, min(200, len(chunks))):
        ids = tok.encode(chunk, add_special_tokens=True)
        if not (10 <= len(ids) <= SEQ_LEN): continue
        b = bb.build_batch([ids])
        o = model(b["input_ids"].to(device), labels=b["labels"].to(device))
        vloss += o["loss"].item(); vn += 1
    avg = vloss / max(vn, 1)
    ppl = math.exp(min(avg, 20))
    print(f"Perplexity on held-out: {ppl:.2f} (random: {V}) | {V/ppl:.1f}x improvement", flush=True)

# Save
torch.save(model.state_dict(), "output/model_large.pt")
torch.save({
    "vocab_size": V, "d_model": 96, "n_heads": 4, "d_ff": 256,
    "n_layers": 4, "max_seq_len": SEQ_LEN, "dropout": 0.3,
    "activation": "silu", "norm_type": "rmsnorm", "pre_norm": True,
    "padding_idx": tok.pad_token_id,
}, "output/model_large_config.pt")
print(f"Saved to output/model_large.pt", flush=True)

# Test generation
print(f"\nSample generations:", flush=True)
for prompt in ["To be, or not to be", "The king", "It was a dark", "Once upon a time"]:
    ids = tok.encode(prompt, add_special_tokens=False)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        gen = model.generate(inp, max_new_tokens=40, temperature=0.5,
            top_k=10, top_p=0.85, eos_token_id=tok.eos_token_id,
            repetition_penalty=1.2)
    text = tok.decode(gen[0].tolist(), skip_special_tokens=True)
    extra = text[len(prompt):].strip()
    print(f"  '{prompt}' -> '{extra[:50]}'", flush=True)
