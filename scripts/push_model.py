"""Publish a trained checkpoint to the HuggingFace Hub.

    export HF_TOKEN=hf_...
    python scripts/push_model.py --repo yourname/myai-xl \
        --checkpoint output/ckpt_900m/checkpoint_latest.pt \
        --tokenizer output/tokenizer.json

Run this from wherever the model was trained - a Kaggle or Colab runtime loses
local files when the session ends, so the checkpoint needs a permanent home
before the machine goes away. Afterwards every script in this project can load
it by address instead of by path:

    python scripts/chat.py --checkpoint hf://yourname/myai-xl/checkpoint_latest.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Target repo, e.g. yourname/myai-xl")
    parser.add_argument("--checkpoint", default="output/checkpoints/checkpoint_latest.pt")
    parser.add_argument("--tokenizer", default="output/tokenizer.json")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--message", default="Add trained checkpoint")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("Set HF_TOKEN first (Settings -> Access Tokens on huggingface.co, "
              "write scope).", file=sys.stderr)
        sys.exit(1)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    for path in (args.checkpoint, args.tokenizer):
        if not os.path.exists(path):
            print(f"Missing: {path}", file=sys.stderr)
            sys.exit(1)

    api = HfApi(token=token)
    api.create_repo(args.repo, private=args.private, exist_ok=True)

    for path in (args.checkpoint, args.tokenizer):
        size = os.path.getsize(path) / 1e9
        print(f"  uploading {os.path.basename(path)} ({size:.2f} GB)...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=os.path.basename(path),
            repo_id=args.repo,
            commit_message=args.message,
        )

    name = os.path.basename(args.checkpoint)
    print(f"\n  https://huggingface.co/{args.repo}")
    print(f"\nUse it anywhere:")
    print(f"  python scripts/chat.py --checkpoint hf://{args.repo}/{name} \\")
    print(f"      --tokenizer output/tokenizer.json")
    if args.private:
        print("\n  Private repo: readers need HF_TOKEN set.")


if __name__ == "__main__":
    main()
