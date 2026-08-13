"""Talk to your AI - built from scratch, trained on Shakespeare."""
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

MODEL_CONFIG = dict(
    vocab_size=V, d_model=128, n_heads=4, d_ff=384,
    n_layers=5, max_seq_len=256, dropout=0.2,
    activation="silu", norm_type="rmsnorm", pre_norm=True,
    padding_idx=tok.pad_token_id,
)
model_path = "output/model.pt"
config_path = "output/model_config.pt"

if os.path.exists(model_path) and os.path.exists(config_path):
    print("Loading model...", end=" ", flush=True)
    cfg = torch.load(config_path, map_location=device, weights_only=False)
    model = LanguageModel(LMConfig(**cfg)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("done! (%s params)" % f"{model.num_parameters():,}", flush=True)
else:
    print("Training model from scratch...", flush=True)
    model = LanguageModel(LMConfig(**MODEL_CONFIG)).to(device)
    print(f"  Model: {model.num_parameters():,} params", flush=True)
    opt = AdamW(model.parameters(), lr=8e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.05)
    sched = WarmupCosineLR(opt, warmup_steps=200, total_steps=4000, min_lr_ratio=0.05)
    clip = GradientClipper(max_norm=1.0)
    bb = BatchBuilder(pad_token_id=tok.pad_token_id, max_length=256, device=device)
    with open("data/tinyshakespeare.txt") as f:
        lines = [l.strip() for l in f if l.strip()]
    random.seed(42)
    torch.manual_seed(42)
    model.train()
    start = time.time()
    for step in range(3000):
        line = random.choice(lines)
        ids = tok.encode(line, add_special_tokens=True)
        if not (10 <= len(ids) <= 256): continue
        batch = bb.build_batch([ids])
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"], labels=batch["labels"])
        loss = out["loss"]
        opt.zero_grad(); loss.backward(); clip.clip(model.parameters())
        opt.step(); sched.step()
        if step % 500 == 499:
            print(f"  step {step+1:4d} | loss {loss.item():.4f} | {time.time()-start:.0f}s", flush=True)
    elapsed = time.time() - start
    print(f"  done: {step+1} steps in {elapsed:.0f}s", flush=True)
    torch.save(model.state_dict(), model_path)
    torch.save(MODEL_CONFIG, config_path)
    model.eval()

print()
print("=" * 55)
print("  TALK TO YOUR AI - BUILT FROM SCRATCH")
print("=" * 55)
print(f"  Model: {model.num_parameters():,} parameters")
print(f"  Trained on: TinyShakespeare (32K lines)")
print(f"  Type a prompt and the model continues it.")
print(f"  Type 'quit' to exit.")
print()

def generate(prompt, max_new=60, temp=0.6, top_k=10, rep=1.2):
    ids = tok.encode(prompt, add_special_tokens=False)
    if not ids: return prompt
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        gen = model.generate(inp, max_new_tokens=max_new, temperature=temp,
            top_k=top_k, top_p=0.85, eos_token_id=tok.eos_token_id,
            repetition_penalty=rep)
    text = tok.decode(gen[0].tolist(), skip_special_tokens=True)
    return text.strip()

while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); break

    if not user_input or user_input.lower() in ("quit", "exit", "q"):
        break

    # Try multiple temperatures and pick the longest response
    best = ""
    for temp in [0.3, 0.5, 0.7]:
        text = generate(user_input, max_new=50, temp=temp, top_k=8, rep=1.3)
        extra = text[len(user_input):].strip()
        if len(extra) > len(best):
            best = text

    if len(best) > len(user_input) + 3:
        print(f"AI:  {best}")
    else:
        text = generate(user_input, max_new=50, temp=0.5, top_k=5, rep=1.5)
        if len(text) > len(user_input):
            print(f"AI:  {text}")
        else:
            print(f"AI:  ...")
    print()
