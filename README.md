# myai — a language model built from scratch

A complete GPT-style language model implemented from first principles in Python.
Tokenizer, transformer, optimizers, training loop, KV-cached inference,
checkpointing and evaluation are all written by hand. PyTorch supplies tensors,
autograd and BLAS — nothing else. There is no `torch.nn.Linear`, no
`torch.nn.MultiheadAttention`, no `torch.optim`, no HuggingFace.

```
143 tests passing · ~9,000 lines · CPU and GPU
```

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python scripts/build_tokenizer.py --type char
```

```bash
python scripts/train.py --preset tiny --steps 2000
```

```bash
python scripts/chat.py                 # terminal
python scripts/gui.py                  # desktop window (pip install PyQt6)
```

Put any `.txt` files in `data/` first — the training script reads everything it
finds there. The repository ships without a corpus; TinyShakespeare and the
Project Gutenberg books used during development are excluded by `.gitignore`.

## Architecture

The default model is a modern decoder-only transformer:

| Component | Choice | Why |
|---|---|---|
| Position | **RoPE** | Relative distance falls out of the dot product; extrapolates better than absolute encodings |
| Attention | **Grouped-query (GQA)** | Fewer KV heads shrinks the cache proportionally at nearly no quality cost |
| Projections | **Fused QKV** | One GEMM instead of three |
| FFN | **SwiGLU** | Gated activation; standard in LLaMA, PaLM, Qwen |
| Normalization | **RMSNorm, pre-norm** | No mean subtraction, no bias; stable gradients at depth |
| Embeddings | **Tied input/output** | Removes a `vocab × d_model` matrix |
| Init | **Depth-scaled** | Residual projections scaled by `1/sqrt(2·n_layers)` so activations stay bounded |
| Decoding | **KV cache** | Generation is O(n) instead of O(n²) |

`ALiBi`, `sinusoidal` and `learned` position encodings are also implemented and
selectable, as are LayerNorm, post-norm, standard multi-head attention and
ReLU/GELU/SiLU feed-forwards.

```python
from myai.nn import LanguageModel, LMConfig

model = LanguageModel(LMConfig(
    vocab_size=8192, d_model=512, n_heads=8, n_kv_heads=2,
    n_layers=8, d_ff=2048, max_seq_len=1024,
    position_encoding="rope", activation="swiglu", tie_embeddings=True,
))

out = model(input_ids, labels=input_ids)
out["loss"].backward()

text = model.generate(prompt_ids, max_new_tokens=100,
                      temperature=0.8, top_k=40, min_p=0.05)
```

### Model presets

| Preset | d_model | Layers | Heads (KV) | Context |
|---|---|---|---|---|
| `tiny` | 128 | 4 | 4 (2) | 256 |
| `mini` | 256 | 6 | 4 (2) | 512 |
| `small` | 512 | 8 | 8 (4) | 1024 |
| `base` | 768 | 12 | 12 (4) | 1024 |

## Performance

Every optimization is measurable — `python scripts/benchmark.py` reports them:

- **Fused optimizers.** Updates run through `torch._foreach_*`, so one call
  covers every parameter tensor instead of a Python loop over hundreds.
- **KV cache.** Preallocated per layer; generating N tokens costs N forward
  passes over one token, not N passes over the whole prefix.
- **Selective LM head.** During generation only the final position is projected
  to vocabulary size, skipping the model's most expensive layer for positions
  whose logits are discarded.
- **Vectorized sampling.** Repetition penalty, top-k, top-p and min-p are batched
  tensor ops with no Python loop over batch or vocabulary.
- **Token packing and length bucketing.** Sequences are concatenated and cut into
  full-length windows (zero padding), or grouped by length when padding is needed.
- **Cached causal mask** and **fused `addmm`** in `Linear`.
- **Optional fused SDPA** (`--flash`) and **`torch.compile`** (`--compile`).
- **Gradient checkpointing** trades ~30% step time for a large drop in
  activation memory.

The hand-written attention path is the default so the project stays honest about
being from scratch. `model.set_flash_attention(True)` switches to PyTorch's fused
kernels; a test asserts both paths produce identical outputs.

A note on what to expect: fused SDPA and `torch.compile` are **GPU**
optimizations. On CPU with a small model they are break-even or slightly slower
(measured 0.93x for SDPA on the `tiny` preset), because there is no memory-bandwidth
wall to win back. The KV cache, fused optimizers, packing and vectorized sampling
help everywhere. Gradient checkpointing costs ~28% step time by design.

Measured on a 4-thread CPU, `tiny` preset, batch 4 × 256 tokens:

```
Training step        754 ms   (1,359 tokens/s)
  fused SDPA         808 ms   (0.93x — GPU-oriented, no CPU win)
  grad checkpoint   1048 ms   (0.72x, large activation-memory saving)
Generation (48 tok)  916 ms with KV cache vs 1162 ms without (1.27x;
                     the gap widens with longer contexts)
```

## Layout

```
myai/
  nn/           Module, Linear, Embedding, RMSNorm, attention, RoPE, transformer, model
  tokenizer/    BPE and character tokenizers, vocabulary, trainer
  data/         ingestion, cleaning, filtering, segmentation, packing, streaming, batching
  train/        optimizers, schedulers, gradient clipping, mixed precision, loop, Trainer
  inference/    KV-cached generation, samplers, streaming, conversation
  checkpoint/   versioned save/load with rotation
  evaluate/     perplexity, accuracy, benchmarks
  config/       schema-validated configuration with presets
  tests/        143 tests
scripts/        build_tokenizer.py · train.py · chat.py · gui.py · benchmark.py
```

## Tokenizers

Both are lossless: `decode(encode(text)) == text`, including newlines and
repeated spaces. Whitespace is carried inside tokens as the `▁` marker, the same
convention SentencePiece uses.

```bash
python scripts/build_tokenizer.py --type char              # ~70 tokens, no OOV
python scripts/build_tokenizer.py --type bpe --vocab-size 4096
```

BPE training uses an incremental pair index, so each merge only rewrites the
words that actually contained the merged pair.

## Training

```bash
python scripts/train.py --preset small --steps 20000 --batch-size 16 \
    --grad-accum 4 --mixed-precision --flash
```

Checkpoints rotate in `output/checkpoints/`, carry the optimizer, scheduler, loop
and RNG state, and resume exactly:

```bash
python scripts/train.py --resume output/checkpoints/checkpoint_latest.pt
```

Pass held-out files to evaluate during training and track the best model:

```bash
python scripts/train.py --preset small --steps 20000 --data data/train --val-data data/val
```

Rotation keeps the last N step files. `checkpoint_latest.pt` always mirrors the
most recent save; with `--val-data`, `checkpoint_best.pt` mirrors the lowest
validation loss seen. Both are byte copies rather than pointers, so the
best model survives even after the step file it came from is rotated away — which
is the normal case, since the best model is rarely among the last N.

The library API is equivalent:

```python
from myai.config.presets import preset_config
from myai.train.engine import Trainer
from myai.data.streaming import StreamingDataset
from myai.tokenizer import load_tokenizer

tokenizer = load_tokenizer("output/tokenizer.json")
config = preset_config("small", max_steps=20000)
config.model.vocab_size = tokenizer.vocab_size

dataset = StreamingDataset(["data"], tokenizer, max_seq_len=1024)
Trainer(config, tokenizer=tokenizer).train(dataset)
```

## Chatting with the model

```bash
python scripts/chat.py                     # terminal
python scripts/gui.py                      # desktop window
python scripts/gui.py --checkpoint output/checkpoints/checkpoint_best.pt
```

Both load the newest checkpoint by default and stream token by token. The desktop
window needs `pip install PyQt6` — nothing else in the project imports it. It puts
the transcript on the left and the sampling controls on the right, so temperature,
top-k, top-p, min-p, repetition penalty and length can be changed between turns
without restarting. Enter sends, Esc stops generation mid-stream, Ctrl+L clears.
Generation runs on a worker thread, so the window stays responsive on CPU.

These are base models trained on plain text — they continue a prompt rather than
answer it. "To be, or" works; "What is the capital of France?" does not.

## Tests

```bash
python -m pytest myai/tests -q
```

Beyond shape checks, the suite pins the properties that are easy to get
silently wrong:

- the model is **not** permutation-invariant (a model without position encoding
  is a bag of words — the control case is tested too)
- RoPE attention scores depend only on relative distance
- KV-cached decoding matches full recomputation exactly
- fused SDPA matches the hand-written attention
- gradient checkpointing produces identical gradients
- moved parameters stay autograd leaves, so gradients survive a device change
- tied weights are one shared parameter, updated once
- tokenizer round-trips are lossless
- `Trainer` trains, checkpoints, and resumes to identical weights
- the best-validation checkpoint survives rotation of the step files
- loss actually decreases

## Requirements

Python 3.10+, PyTorch 2.0+. CPU works; CUDA is used automatically when available.

## License

MIT
