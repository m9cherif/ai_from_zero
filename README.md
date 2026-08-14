# myai — a language model built from scratch

A complete GPT-style language model implemented from first principles in Python.
Tokenizer, transformer, optimizers, training loop, KV-cached inference,
checkpointing and evaluation are all written by hand. PyTorch supplies tensors,
autograd and BLAS — nothing else. There is no `torch.nn.Linear`, no
`torch.nn.MultiheadAttention`, no `torch.optim`, no HuggingFace.

```
173 tests passing · ~12,900 lines · CPU and GPU
```

## Quick start

```bash
pip install -r requirements.txt
```

```bash
python scripts/fetch_corpus.py                                    # ~90 MB of real text
python scripts/build_tokenizer.py --type bpe --vocab-size 4096 --data 'data/train/*.txt'
python scripts/train.py --preset mini --steps 5000 --data data/train --val-data data/val
python scripts/chat.py                 # terminal
python scripts/gui.py                  # desktop window (pip install PyQt6)
```

The repository ships without a corpus — `scripts/fetch_corpus.py` downloads one.
Any `.txt` files you drop in `data/` are picked up too.

## Data

`scripts/fetch_corpus.py` assembles a multi-domain corpus from sources that stay
reachable behind restrictive proxies (everything comes from
`raw.githubusercontent.com`, so it works where `huggingface.co` and
`gutenberg.org` are blocked):

| Source | Content |
|---|---|
| Project Gutenberg (GITenberg mirror) | ~100 works — Melville, Austen, Dickens, Dostoevsky, Tolstoy, Verne, Conrad, Shakespeare, Milton, Plato, Kant, Darwin, Einstein |
| WikiText-2 | curated Wikipedia, the standard LM benchmark corpus |
| Norvig `big.txt` | mixed reference prose |

Gutenberg licence headers and footers are stripped, and validation documents are
held out **whole**, so no text appears on both sides of the split. Titles that
have moved or been renamed in the mirror are skipped rather than failing the run,
so the exact total varies a little:

```
101 documents    89.6 MB cleaned
train            84.2 MB   24,866,609 tokens
val               5.4 MB    1,854,988 tokens (held out whole)
```

Tokenizing that corpus on every epoch costs more CPU than the model does, so
`--cache-dir` tokenizes once into a memory-mapped array of ids and trains from
that. Windows are exactly `max_seq_len`, so nothing is padded:

```bash
python scripts/train.py --cache-dir output/tokens --data data/train --val-data data/val
```

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
| `xl` | 1536 | 28 | 12 (4) | 1024 |

`xl` is ~982M parameters. The architecture scales to it unchanged, but fp32
weights, gradients and the two AdamW moments come to **15.7 GB before a single
activation**, so it needs a 40 GB+ accelerator — this is a hardware step change,
not a longer wait. [`docs/SCALING.md`](docs/SCALING.md) has the measured
arithmetic, the extrapolated training time, and why a 1B model is the wrong
choice for a small token budget regardless of hardware.

## Hardware selection

`--device auto` (the default) resolves the best available compute, and every
entry point routes through `myai/core/device.py`. Two failure modes it exists to
avoid, both specific to servers:

**The wrong GPU.** `torch.cuda.is_available()` only says *whether* a GPU exists;
everything then lands on `cuda:0`, which on a shared box is often the card
someone else is already filling. Selection is by **free** memory, so a run does
not OOM on a machine that had 38 GB idle on another card.

**The wrong thread count.** `os.cpu_count()` reports the host's cores, not the
container's share. A pod limited to 2 CPUs on a 64-core host still sees 64, so
PyTorch starts 64 threads that contend over 2 cores worth of runtime. The thread
pool is sized from the **cgroup quota** (v1 and v2) intersected with the CPU
**affinity mask** — what the process may actually use.

```bash
python scripts/train.py --device auto       # emptiest GPU, else MPS, else CPU
python scripts/train.py --device cuda:1     # explicit, never redirected
python scripts/train.py --threads 8         # override thread detection
```

```python
from myai.core.device import setup, describe
device = setup("auto")        # selects and sizes threads
print(describe(device))       # cpu, 4 usable CPUs, 4 threads
```

An explicit device is honoured as given — the emptiest-GPU rule applies only to
`auto`, so pinning a run to a particular card always works. Requesting CUDA on a
machine without it warns and falls back to CPU rather than crashing.

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

A note on what to expect: the KV cache, fused optimizers, packing and vectorized
sampling help everywhere. Fused SDPA is hardware-dependent — it is a large win on
GPU, and on CPU it ranges from a slight loss to a solid win depending on the
machine (0.93x on one 4-core box, 1.34x on another). Measure it on your own
hardware rather than trusting either number. Gradient checkpointing costs step
time by design, in exchange for activation memory.

Measured on 4 Xeon cores @ 2.80GHz, `tiny` preset, batch 8 × 256 tokens:

```
Training step        106 ms   (19,251 tokens/s)
  fused SDPA          80 ms   (1.34x)
  grad checkpoint    121 ms   (0.88x, large activation-memory saving)
Generation (64 tok)  117 ms with KV cache vs 197 ms without (1.69x;
                     the gap widens with longer contexts)
```

Training throughput by preset, same machine, batch 8 × 256, vocab 2048:

| Preset | Params | Steps/s | Tokens/s |
|---|---|---|---|
| `tiny` | 1.05M | 7.3 | 14,992 |
| `mini` | 4.95M | 3.1 | 6,245 |
| `small` | 24.7M | 0.90 | 1,835 |
| `base` | 77.1M | 0.33 | 675 |

Steps/s is not a property of the hardware alone — a step is whatever
`batch × seq_len` you choose. On the `tiny` preset, batch 1 gives 30 steps/s and
batch 32 gives 2.5, while tokens/s *rises* from 7,716 to 20,277 across that same
range because larger batches amortize per-step overhead. Compare configurations
by tokens/s; use steps/s only to estimate wall clock for a fixed step budget.

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
  tests/        173 tests
scripts/        fetch_corpus.py · build_tokenizer.py · train.py · evaluate.py · chat.py · gui.py · benchmark.py
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

## A trained model

An 8.3M-parameter model trained end to end on the corpus above, on four CPU
cores — no GPU:

| | |
|---|---|
| Parameters | 8,319,040 (d_model 320, 6 layers, GQA 4/2, vocab 4,096) |
| Corpus | 24,866,609 tokens, one full epoch |
| Steps | 6,071 at batch 16 × 256 |
| Wall clock | 2h 11m at ~3,200 tokens/s |
| **Held-out perplexity** | **47.80** (uniform baseline 4,096) |

Perplexity is measured with `scripts/evaluate.py` on 408,000 tokens from five
books that are in neither the training nor validation split — verified at 0/300
overlapping probes, because a number measured on contaminated text is not a
number.

```
'To be, or not'         -> 'to be done; but this is the first / Would not I tell you,
                            which is a kind of friend, / Excellent and unfortuna'
'She looked at him and' -> 'she turned to her firm, as though he had a handsome
                            smile; but she was not quite represented by the reproa'
'In the beginning'      -> 'of the poop, / As a winter, the great city, the first-rate
                            of the / Breath-day, the cellar-shops, the two, and'
```

It completes *"To be, or not to be"*, inflects verbs correctly, keeps quotes and
clauses balanced, and shifts register between verse and prose depending on the
prompt. It is a base model at 8M parameters: it continues text, it does not
answer questions, and it will not stay on topic for long.

Worth knowing: `checkpoint_best.pt` scored **50.65** on that same held-out set
while the final weights scored **47.80**. The best checkpoint is only as good as
its last evaluation, and training rarely stops on an eval boundary. A final
evaluation now runs when training ends, so the two cannot drift apart this way.

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
