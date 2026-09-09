# Scaling: what it takes to train 1B parameters

This document records what a 1-billion-parameter run actually costs, measured on
the machine this project was developed on rather than estimated from a blog post.
The `xl` preset in `myai/config/presets.py` is that model. It is a real,
instantiable config — the architecture scales to it without changes — but it does
not fit on a CPU, and the reason is worth writing down.

## The model

```python
from myai.config.presets import preset_config
config = preset_config("xl")          # d_model 1536, 28 layers, d_ff 6144, GQA 12/4
config.model.vocab_size = 8192
```

```
parameters: 981,554,688  (0.982B)
```

## Why it does not run on CPU

The blocker is memory, and it is a step change rather than a longer wait.
AdamW keeps two fp32 moments per parameter, and the gradient buffer matches the
weights:

| Buffer | Size |
|---|---|
| weights (fp32) | 3.93 GB |
| gradients | 3.93 GB |
| Adam `m` | 3.93 GB |
| Adam `v` | 3.93 GB |
| **total** | **15.70 GB** |

That is the floor *before a single activation is allocated*. A 16 GB machine
cannot hold it, and neither can the 15 GB development box — the process dies
during optimizer construction, not partway through training.

Speed is the second wall. Measured throughput on 4 Xeon cores at batch 8 × 256:

| Params | Tokens/s | Steps/s |
|---|---|---|
| 1.05M | 14,992 | 7.3 |
| 4.95M | 6,245 | 3.1 |
| 24.7M | 1,835 | 0.90 |
| 77.1M | 675 | 0.33 |

Extrapolating the measured 77M number linearly — which is optimistic, since
larger models lose cache locality — gives **~53 tokens/s** for the 1B model.
A Chinchilla-optimal budget for 1B parameters is roughly 20B tokens:

```
20e9 tokens / 53 tokens/s = 3.8e8 seconds = 11.7 years
```

Four hours of that run would cover 0.76M tokens: **0.0039%** of the budget.

## What the hardware would need to be

- **Memory:** a 40 GB+ accelerator (A100 40/80GB, H100) for the 15.7 GB of
  optimizer state plus activations, or multiple smaller GPUs with the optimizer
  state sharded (ZeRO-2/3, FSDP).
- **Throughput:** on one A100 this class of model trains in the low thousands of
  tokens/s; 20B tokens is on the order of weeks. Published 1B runs use dozens of
  GPUs for days.
- **Data:** 20B tokens is roughly 80 GB of clean text — FineWeb, The Pile, C4.
  Not something to assemble file by file.

The command itself is unchanged, which is the point of the config system:

```bash
python scripts/train.py --preset xl --steps 200000 --batch-size 16 \
    --grad-accum 8 --mixed-precision --flash --device cuda \
    --data data/train --val-data data/val
```

## The compute-optimal counterargument

Even with the memory available, 1B is the wrong model for a small token budget.
Chinchilla's result is that compute is best spent when tokens ≈ 20 × parameters.
Inverting it for the budget actually available here:

| Wall clock | Tokens reachable | Compute-optimal model |
|---|---|---|
| 1 hour | ~25M | ~1.2M params |
| 1 day | ~600M | ~30M params |
| 1 month | ~18B | ~900M params |

A 1B model trained on 25M tokens is not a better model than a 5M model trained
on the same 25M tokens — it is a worse one, because almost none of its capacity
gets used and every step costs 200× more. Parameter count is a budget to be
matched to the data, not a target to be maximized.

This is why the model actually trained here is sized to its corpus. See the
README for that run.
