"""Publish a trained model to the HuggingFace Hub.

    export HF_TOKEN=hf_...
    python scripts/push_model.py --checkpoint output/ckpt_900m/checkpoint_latest.pt

Uploads the weights, the tokenizer, a config.json and a model card. The card is
generated from the checkpoint rather than written by hand, so the parameter
count and geometry it advertises cannot drift from the file beside it.

Run this from wherever the model was trained: a Kaggle or Colab runtime deletes
local files when the session ends, and a multi-gigabyte checkpoint cannot live
in a git repository. Afterwards every entry point takes the address:

    python scripts/chat.py --checkpoint hf://owner/repo/checkpoint_latest.pt
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_REPO = "m9cherif3/ai_from_scratch"


def read_checkpoint_facts(path: str) -> dict:
    """Geometry and training position, straight from the file being uploaded."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("config", {})
    model = raw.get("model", raw) if isinstance(raw, dict) else {}
    loop = checkpoint.get("loop_state") or {}

    params = sum(v.numel() for v in checkpoint["model_state_dict"].values())
    # Tied embeddings are stored once but counted twice by a naive sum.
    if model.get("tie_embeddings", model.get("weight_tying", True)):
        if "lm_head.weight" in checkpoint["model_state_dict"]:
            params -= checkpoint["model_state_dict"]["lm_head.weight"].numel()

    return {
        "parameters": params,
        "d_model": model.get("d_model"),
        "n_layers": model.get("n_layers"),
        "n_heads": model.get("n_heads"),
        "n_kv_heads": model.get("n_kv_heads"),
        "d_ff": model.get("d_ff"),
        "max_seq_len": model.get("max_seq_len"),
        "vocab_size": model.get("vocab_size"),
        "position_encoding": model.get("position_encoding", "rope"),
        "activation": model.get("activation", "swiglu"),
        "norm_type": model.get("norm_type", "rmsnorm"),
        "step": loop.get("global_step"),
        "best_loss": loop.get("best_loss"),
    }


def model_card(repo: str, facts: dict, checkpoint_name: str) -> str:
    step = f"{facts['step']:,}" if facts.get("step") else "unknown"
    params = facts["parameters"]
    return f"""---
license: mit
library_name: myai
tags:
  - text-generation
  - from-scratch
  - pytorch
pipeline_tag: text-generation
---

# ai_from_scratch — {params/1e6:.1f}M parameters

A GPT-style language model written from first principles. The tokenizer,
transformer, optimizers, training loop, KV-cached inference and checkpointing
are all hand-written; PyTorch supplies tensors, autograd and BLAS and nothing
else. There is no `torch.nn.Linear`, no `torch.optim`, no `transformers`.

Code: <https://github.com/m9cherif/ai_from_zero> (branch `main-5h8bvh`)

## Architecture

| | |
|---|---|
| Parameters | **{params:,}** |
| Layers | {facts['n_layers']} |
| d_model | {facts['d_model']} |
| Heads (KV) | {facts['n_heads']} ({facts['n_kv_heads']}) — grouped-query attention |
| d_ff | {facts['d_ff']} |
| Context | {facts['max_seq_len']} tokens |
| Vocabulary | {facts['vocab_size']:,} (byte-pair encoding) |
| Position | {facts['position_encoding'].upper()} |
| Activation | {facts['activation']} |
| Normalization | {facts['norm_type']}, pre-norm |
| Embeddings | tied input/output |
| Training step | {step} |

## Usage

```bash
git clone -b main-5h8bvh https://github.com/m9cherif/ai_from_zero
cd ai_from_zero && pip install -r requirements.txt

python scripts/chat.py --checkpoint hf://{repo}/{checkpoint_name} \\
    --tokenizer hf://{repo}/tokenizer.json --prompt "The old man walked into the"
```

The same address works for `evaluate.py`, `serve.py` (an HTTP endpoint) and
`train.py --resume`, which is how training continues across sessions on
time-limited hardware.

## What it is

A **base** model. It continues text; it does not answer questions or follow
instructions. `"To be, or not"` works; `"What is the capital of France?"`
produces more question-shaped text rather than an answer.

Training data is public-domain literature (Project Gutenberg), curated
Wikipedia and open web text. Validation documents are held out whole and the
corpus builder measures its own train/validation overlap, so reported
perplexity is not inflated by leakage.

## Honest limitations

Model quality is bounded by tokens seen, not by parameter count. A model of
this size wants roughly 20 tokens per parameter to train properly; anything
trained far below that has learned which words are common and little more.
Check the training step above against that ratio before expecting fluency.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Default: {DEFAULT_REPO}")
    parser.add_argument("--checkpoint", default="output/checkpoints/checkpoint_latest.pt")
    parser.add_argument("--tokenizer", default="output/tokenizer.json")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--message", default="Update trained checkpoint")
    parser.add_argument("--no-card", action="store_true", help="Leave the model card alone")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the card and the upload plan without uploading")
    args = parser.parse_args()

    for path in (args.checkpoint, args.tokenizer):
        if not os.path.exists(path):
            print(f"Missing: {path}", file=sys.stderr)
            sys.exit(1)

    facts = read_checkpoint_facts(args.checkpoint)
    name = os.path.basename(args.checkpoint)
    print(f"  {facts['parameters']:,} parameters, step {facts['step']}")

    staging = os.path.join(os.path.dirname(args.checkpoint) or ".", "_hub")
    os.makedirs(staging, exist_ok=True)
    config_path = os.path.join(staging, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(facts, f, indent=2)

    uploads = [(args.checkpoint, name),
               (args.tokenizer, "tokenizer.json"),
               (config_path, "config.json")]

    if not args.no_card:
        card_path = os.path.join(staging, "README.md")
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(model_card(args.repo, facts, name))
        uploads.append((card_path, "README.md"))

    if args.dry_run:
        print(f"\nWould upload to https://huggingface.co/{args.repo}:")
        for local, remote in uploads:
            print(f"  {os.path.getsize(local)/1e6:9.1f} MB  {remote}")
        print("\n--- model card ---")
        print(model_card(args.repo, facts, name))
        return

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("Set HF_TOKEN (huggingface.co -> Settings -> Access Tokens, write "
              "scope). On Kaggle use Add-ons -> Secrets.", file=sys.stderr)
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    api.create_repo(args.repo, private=args.private, exist_ok=True, repo_type="model")

    for local, remote in uploads:
        print(f"  uploading {remote} ({os.path.getsize(local)/1e6:.1f} MB)...")
        api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                        repo_id=args.repo, commit_message=args.message)

    print(f"\n  https://huggingface.co/{args.repo}")
    print("\nUse it anywhere:")
    print(f"  python scripts/chat.py --checkpoint hf://{args.repo}/{name} \\")
    print(f"      --tokenizer hf://{args.repo}/tokenizer.json")


if __name__ == "__main__":
    main()
