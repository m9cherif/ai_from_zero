"""End-to-end integration tests.

These exercise the paths that unit tests never touched - Trainer,
TrainingLoop, StreamingDataset, InferenceEngine - which is exactly where the
crashes lived: a state_dict called as a method that was a property, a missing
import, and a generator function that never returned text.
"""

import math
from pathlib import Path

import pytest
import torch

from ..config.presets import TrainConfig, preset_config
from ..train.engine import Trainer, build_model_config
from ..train.loop import TrainingLoop, iter_batches
from ..train.optimizer import AdamW
from ..data.streaming import StreamingDataset
from ..inference import InferenceEngine, AutoregressiveGenerator
from ..tokenizer import CharTokenizer, TokenizerTrainer, load_tokenizer
from ..nn.model import LanguageModel, LMConfig


CORPUS = (
    "the quick brown fox jumps over the lazy dog.\n"
    "a language model learns to predict the next token.\n"
    "\n"
    "training from scratch means building every component by hand.\n"
    "attention is a weighted sum of values.\n"
) * 40


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    (tmp_path / "corpus.txt").write_text(CORPUS, encoding="utf-8")
    return tmp_path


@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer.train([CORPUS])


def make_config(corpus_dir: Path, tmp_path: Path, vocab_size: int, **overrides) -> TrainConfig:
    config = TrainConfig(
        model={
            "d_model": 32, "n_heads": 4, "n_kv_heads": 2, "n_layers": 2,
            "d_ff": 64, "max_seq_len": 32, "dropout": 0.0, "vocab_size": vocab_size,
        },
        data={"data_paths": [str(corpus_dir)], "batch_size": 4, "max_seq_len": 32},
        optimizer={"learning_rate": 1e-3, "gradient_accumulation_steps": 1},
        scheduler={"scheduler": "warmup_cosine", "warmup_steps": 2},
        checkpoint={"save_dir": str(tmp_path / "checkpoints"), "save_every_steps": 1000},
        logging={"log_every_steps": 1000},
        hardware={"device": "cpu"},
        max_steps=6,
        num_epochs=1,
        **overrides,
    )
    return config


class TestStreamingDataset:
    def test_yields_many_sequences_from_one_file(self, corpus_dir, tokenizer):
        """A .txt file arrives as one string; it must not become one sample."""
        dataset = StreamingDataset(
            data_paths=[str(corpus_dir)], tokenizer=tokenizer,
            max_seq_len=32, shuffle_buffer_size=4,
        )
        sequences = dataset.take(20)
        assert len(sequences) == 20
        assert all(len(s) > 1 for s in sequences)

    def test_packing_fills_sequences_completely(self, corpus_dir, tokenizer):
        dataset = StreamingDataset(
            data_paths=[str(corpus_dir)], tokenizer=tokenizer,
            max_seq_len=32, shuffle_buffer_size=2, pack_sequences=True,
        )
        sequences = dataset.take(10)
        # Packed sequences are exactly max_seq_len - zero padding waste.
        assert all(len(s) == 32 for s in sequences)

    def test_unpacked_mode_respects_max_len(self, corpus_dir, tokenizer):
        dataset = StreamingDataset(
            data_paths=[str(corpus_dir)], tokenizer=tokenizer,
            max_seq_len=16, shuffle_buffer_size=2, pack_sequences=False,
        )
        assert all(len(s) <= 16 for s in dataset.take(10))


class TestBatching:
    def test_batch_size_is_honoured(self):
        data = [[1] * (i % 7 + 2) for i in range(40)]
        batches = list(iter_batches(data, batch_size=4, length_bucketing=False))
        assert all(len(b) == 4 for b in batches[:-1])
        assert sum(len(b) for b in batches) == 40

    def test_length_bucketing_groups_similar_lengths(self):
        data = [[1] * length for length in [1, 50, 2, 60, 3, 70, 4, 80]]
        batches = list(iter_batches(data, batch_size=2, length_bucketing=True, bucket_multiplier=4))
        # Sorting by length means each batch pairs short with short.
        spreads = [max(len(s) for s in b) - min(len(s) for s in b) for b in batches]
        assert max(spreads) <= 10

    def test_drop_last(self):
        data = [[1, 2]] * 10
        batches = list(iter_batches(data, batch_size=4, length_bucketing=False, drop_last=True))
        assert all(len(b) == 4 for b in batches)


class TestTrainerEndToEnd:
    def test_trains_saves_and_reports_loss(self, corpus_dir, tmp_path, tokenizer):
        config = make_config(corpus_dir, tmp_path, tokenizer.vocab_size)
        dataset = StreamingDataset(
            data_paths=[str(corpus_dir)], tokenizer=tokenizer, max_seq_len=32,
            shuffle_buffer_size=4,
        )

        trainer = Trainer(config, tokenizer=tokenizer)
        stats = trainer.train(dataset)

        assert trainer._loop.global_step > 0
        assert math.isfinite(stats["loss"])
        assert stats["tokens"] > 0

        saved = trainer._checkpoint_manager.list_checkpoints()
        assert saved, "training must produce a checkpoint"

    def test_best_checkpoint_written_when_validating(self, corpus_dir, tmp_path, tokenizer):
        """save_best is a documented config flag; it has to actually produce a file."""
        config = make_config(corpus_dir, tmp_path, tokenizer.vocab_size, eval_every_steps=2, eval_steps=2)
        dataset = StreamingDataset(
            data_paths=[str(corpus_dir)], tokenizer=tokenizer, max_seq_len=32,
            shuffle_buffer_size=4,
        )
        val_dataset = StreamingDataset(
            data_paths=[str(corpus_dir)], tokenizer=tokenizer, max_seq_len=32,
            shuffle_buffer_size=4,
        )

        trainer = Trainer(config, tokenizer=tokenizer)
        trainer.train(dataset, val_dataset)

        best_path = trainer._checkpoint_manager.best_checkpoint_path()
        assert best_path.exists(), "validation ran, so a best checkpoint must exist"

        restored = trainer._checkpoint_manager.resume_from_best()
        assert restored is not None
        assert math.isfinite(restored["loop_state"]["best_loss"])

    def test_no_best_checkpoint_without_validation(self, corpus_dir, tmp_path, tokenizer):
        config = make_config(corpus_dir, tmp_path, tokenizer.vocab_size)
        dataset = StreamingDataset(
            data_paths=[str(corpus_dir)], tokenizer=tokenizer, max_seq_len=32,
            shuffle_buffer_size=4,
        )

        trainer = Trainer(config, tokenizer=tokenizer)
        trainer.train(dataset)

        assert not trainer._checkpoint_manager.best_checkpoint_path().exists()

    def test_save_best_disabled_writes_no_best(self, corpus_dir, tmp_path, tokenizer):
        config = make_config(corpus_dir, tmp_path, tokenizer.vocab_size, eval_every_steps=2, eval_steps=2)
        config.checkpoint.save_best = False
        dataset = StreamingDataset(
            data_paths=[str(corpus_dir)], tokenizer=tokenizer, max_seq_len=32,
            shuffle_buffer_size=4,
        )
        val_dataset = StreamingDataset(
            data_paths=[str(corpus_dir)], tokenizer=tokenizer, max_seq_len=32,
            shuffle_buffer_size=4,
        )

        trainer = Trainer(config, tokenizer=tokenizer)
        trainer.train(dataset, val_dataset)

        assert not trainer._checkpoint_manager.best_checkpoint_path().exists()

    def test_checkpoint_roundtrip_restores_weights(self, corpus_dir, tmp_path, tokenizer):
        config = make_config(corpus_dir, tmp_path, tokenizer.vocab_size)
        dataset = StreamingDataset(
            data_paths=[str(corpus_dir)], tokenizer=tokenizer, max_seq_len=32,
            shuffle_buffer_size=4,
        )

        trainer = Trainer(config, tokenizer=tokenizer)
        trainer.train(dataset)
        path = trainer._checkpoint_manager.list_checkpoints()[-1]

        reference = {k: v.clone() for k, v in trainer.model.state_dict().items()}
        steps_done = trainer._loop.global_step

        resume_config = make_config(corpus_dir, tmp_path, tokenizer.vocab_size)
        resume_config.checkpoint.resume_from = path
        resumed = Trainer(resume_config, tokenizer=tokenizer)

        assert resumed._loop.global_step == steps_done
        for key, value in resumed.model.state_dict().items():
            assert torch.allclose(value, reference[key]), f"weight {key} did not survive the round trip"

    def test_loss_decreases_on_a_tiny_corpus(self, corpus_dir, tmp_path, tokenizer):
        """The whole stack must actually learn, not merely run."""
        torch.manual_seed(0)
        model = LanguageModel(LMConfig(
            vocab_size=tokenizer.vocab_size, d_model=64, n_heads=4, n_kv_heads=2,
            d_ff=128, n_layers=2, max_seq_len=64, dropout=0.0,
        ))
        optimizer = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

        ids = torch.tensor([tokenizer.encode(CORPUS[:640], add_special_tokens=False)[:64]])
        model.train()

        first = model(ids, labels=ids)["loss"].item()
        for _ in range(30):
            optimizer.zero_grad()
            loss = model(ids, labels=ids)["loss"]
            loss.backward()
            optimizer.step()
        last = loss.item()

        assert last < first * 0.5, f"loss did not fall: {first:.3f} -> {last:.3f}"


class TestInferenceEngine:
    def test_generate_returns_a_string(self, tokenizer):
        """generate() must return text, not a generator object."""
        model = LanguageModel(LMConfig(
            vocab_size=tokenizer.vocab_size, d_model=32, n_heads=4, d_ff=64,
            n_layers=2, max_seq_len=64, dropout=0.0,
        ))
        engine = InferenceEngine(device="cpu")
        engine.load_model(model, tokenizer)

        out = engine.generate("the quick", max_new_tokens=8, temperature=0.8, top_k=5)
        assert isinstance(out, str)
        assert out.startswith("the quick")

    def test_generate_without_prompt_echo(self, tokenizer):
        model = LanguageModel(LMConfig(
            vocab_size=tokenizer.vocab_size, d_model=32, n_heads=4, d_ff=64,
            n_layers=2, max_seq_len=64, dropout=0.0,
        ))
        engine = InferenceEngine(device="cpu")
        engine.load_model(model, tokenizer)

        out = engine.generate("the quick", max_new_tokens=8, return_full_text=False)
        assert isinstance(out, str)
        assert not out.startswith("the quick")

    def test_stream_yields_chunks(self, tokenizer):
        model = LanguageModel(LMConfig(
            vocab_size=tokenizer.vocab_size, d_model=32, n_heads=4, d_ff=64,
            n_layers=2, max_seq_len=64, dropout=0.0,
        ))
        engine = InferenceEngine(device="cpu")
        engine.load_model(model, tokenizer)

        chunks = list(engine.generate_stream("the", max_new_tokens=6, temperature=0.8))
        assert chunks
        assert all(isinstance(c, str) for c in chunks)

    def test_batch_generation(self, tokenizer):
        model = LanguageModel(LMConfig(
            vocab_size=tokenizer.vocab_size, d_model=32, n_heads=4, d_ff=64,
            n_layers=2, max_seq_len=64, dropout=0.0,
        ))
        engine = InferenceEngine(device="cpu")
        engine.load_model(model, tokenizer)

        outs = engine.generate_batch(["the", "a language"], max_new_tokens=5)
        assert len(outs) == 2
        assert all(isinstance(o, str) for o in outs)

    def test_load_from_checkpoint(self, corpus_dir, tmp_path, tokenizer):
        config = make_config(corpus_dir, tmp_path, tokenizer.vocab_size)
        dataset = StreamingDataset(
            data_paths=[str(corpus_dir)], tokenizer=tokenizer, max_seq_len=32,
            shuffle_buffer_size=4,
        )
        trainer = Trainer(config, tokenizer=tokenizer)
        trainer.train(dataset)
        path = trainer._checkpoint_manager.list_checkpoints()[-1]

        engine = InferenceEngine(device="cpu")
        engine.load_checkpoint(path)
        assert engine.model is not None
        assert engine.model.config.d_model == 32

        engine._tokenizer = tokenizer
        engine._generator = AutoregressiveGenerator(engine.model, tokenizer, device=engine.device)
        assert isinstance(engine.generate("the", max_new_tokens=4), str)


class TestTokenizerRoundTrip:
    @pytest.mark.parametrize("text", [
        "hello world",
        "multiple   spaces preserved",
        "line one\nline two\n\nline four",
        "punctuation, and: symbols!",
        "\ttabbed and trailing space ",
    ])
    def test_bpe_roundtrip_is_lossless(self, text):
        trainer = TokenizerTrainer(target_vocab_size=200, min_frequency=1)
        tok = trainer.train(iter([CORPUS, text]), verbose=False)
        assert tok.decode(tok.encode(text, add_special_tokens=False)) == text

    def test_bpe_merges_actually_compress(self):
        trainer = TokenizerTrainer(target_vocab_size=300, min_frequency=2)
        tok = trainer.train(iter([CORPUS]), verbose=False)
        text = "the quick brown fox jumps over the lazy dog."
        assert len(tok.encode(text, add_special_tokens=False)) < len(text) / 2

    def test_char_roundtrip_is_lossless(self):
        tok = CharTokenizer.train([CORPUS])
        text = "the quick\nbrown  fox."
        assert tok.decode(tok.encode(text, add_special_tokens=True)) == text

    def test_load_tokenizer_detects_type(self, tmp_path):
        char_path = tmp_path / "char.json"
        CharTokenizer.train([CORPUS]).save(str(char_path))
        assert isinstance(load_tokenizer(str(char_path)), CharTokenizer)

        bpe_path = tmp_path / "bpe.json"
        TokenizerTrainer(target_vocab_size=150, min_frequency=1).train(
            iter([CORPUS]), verbose=False
        ).save(str(bpe_path))
        from ..tokenizer import BPETokenizer
        assert isinstance(load_tokenizer(str(bpe_path)), BPETokenizer)


class TestConfigPresets:
    @pytest.mark.parametrize("name", ["tiny", "mini", "small", "base"])
    def test_presets_build_valid_models(self, name):
        config = preset_config(name, max_steps=1)
        model_config = build_model_config(config)
        assert model_config.d_model % model_config.n_heads == 0
        assert model_config.n_heads % (model_config.n_kv_heads or model_config.n_heads) == 0

    def test_unknown_preset_rejected(self):
        with pytest.raises(ValueError):
            preset_config("gigantic")

    def test_effective_batch_size(self):
        config = TrainConfig(
            data={"batch_size": 8}, optimizer={"gradient_accumulation_steps": 4}
        )
        assert config.effective_batch_size() == 32
