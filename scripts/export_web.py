"""Export a trained checkpoint for in-browser inference.

    python scripts/export_web.py --out web/model.json

Quantizes every weight matrix to int8 with a per-output-row scale, packs the
whole model plus the tokenizer into one JSON payload, and reports the size.
A browser can then run the model with no server: see scripts/build_web.py,
which inlines this payload into a self-contained page.

Per-row scales matter. A single scale for a whole matrix is set by its largest
outlier, which crushes every other row into a handful of levels; scaling each
output row independently keeps small rows at full resolution for one extra
float per row.
"""

import argparse
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


def quantize_rows(tensor: torch.Tensor):
    """Symmetric int8 quantization with one scale per output row."""
    matrix = tensor.detach().float()
    if matrix.dim() == 1:                       # norm gains stay float
        return None, matrix

    peak = matrix.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = peak / 127.0
    codes = torch.round(matrix / scale).clamp(-127, 127).to(torch.int8)
    return codes, scale.squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="output/checkpoints/checkpoint_latest.pt")
    parser.add_argument("--tokenizer", default="output/tokenizer.json")
    parser.add_argument("--out", default="web/model.json")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    raw_config = checkpoint.get("config", {})
    model_config = raw_config.get("model", raw_config) if isinstance(raw_config, dict) else {}

    tied = bool(model_config.get("tie_embeddings", True))

    blobs = []
    meta = {}
    offset = 0
    kept = 0

    for name, tensor in state.items():
        # The LM head is the embedding matrix when weights are tied; shipping
        # it twice would add 1.3M parameters of pure duplication.
        if tied and name == "lm_head.weight":
            continue

        codes, scale = quantize_rows(tensor)
        entry = {"shape": list(tensor.shape)}
        if codes is None:
            data = scale.numpy().astype("<f4").tobytes()
            entry.update(dtype="f32", offset=offset, bytes=len(data))
            blobs.append(data)
            offset += len(data)
        else:
            qbytes = codes.numpy().tobytes()
            sbytes = scale.numpy().astype("<f4").tobytes()
            entry.update(dtype="q8", offset=offset, bytes=len(qbytes),
                         scale_offset=offset + len(qbytes), scale_bytes=len(sbytes))
            blobs.append(qbytes)
            blobs.append(sbytes)
            offset += len(qbytes) + len(sbytes)
        meta[name] = entry
        kept += tensor.numel()

    blob = b"".join(blobs)
    tokenizer = json.load(open(args.tokenizer, encoding="utf-8"))

    payload = {
        "config": {
            "d_model": model_config["d_model"],
            "n_layers": model_config["n_layers"],
            "n_heads": model_config["n_heads"],
            "n_kv_heads": model_config.get("n_kv_heads") or model_config["n_heads"],
            "vocab_size": model_config["vocab_size"],
            "max_seq_len": model_config["max_seq_len"],
            "rope_base": model_config.get("rope_base", 10000.0),
            "norm_eps": 1e-5,
            "tied": tied,
        },
        "tensors": meta,
        "blob": base64.b64encode(blob).decode("ascii"),
        "tokenizer": {
            "token_to_id": tokenizer["vocab"]["token_to_id"],
            "merges": tokenizer["merges"],
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size = os.path.getsize(args.out)
    print(f"  parameters exported  {kept:,}")
    print(f"  binary blob          {len(blob)/1e6:.2f} MB")
    print(f"  payload (base64)     {size/1e6:.2f} MB  -> {args.out}")
    print(f"  vocab                {len(payload['tokenizer']['token_to_id']):,} tokens, "
          f"{len(payload['tokenizer']['merges']):,} merges")


if __name__ == "__main__":
    main()
