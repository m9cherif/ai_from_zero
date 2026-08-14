"""Preset configurations for different model scales."""

from ..core.types import Device
from .base import Config, ConfigValidator as V


class TokenizerConfig(Config):
    _schema = {
        "vocab_size": {
            "default": 8192,
            "validate": lambda v, n: V.check_positive(v, n),
        },
        "min_frequency": {
            "default": 2,
            "validate": lambda v, n: V.check_positive(v, n),
        },
        "max_token_length": {
            "default": None,
            "validate": lambda v, n: V.check_positive(v, n) if v is not None else None,
        },
        "special_tokens": {
            "default": {
                "pad": "[PAD]",
                "unk": "[UNK]",
                "bos": "[BOS]",
                "eos": "[EOS]",
                "mask": "[MASK]",
            },
        },
        "tokenizer_type": {
            "default": "bpe",
            "validate": lambda v, n: V.check_in(v, ["bpe", "word", "character"], n),
        },
    }


class ModelConfig(Config):
    _schema = {
        "d_model": {
            "default": 256,
            "validate": lambda v, n: V.check_positive(v, n),
        },
        "n_heads": {
            "default": 8,
            "validate": lambda v, n: V.check_positive(v, n),
        },
        "n_layers": {
            "default": 6,
            "validate": lambda v, n: V.check_positive(v, n),
        },
        "d_ff": {
            "default": 1024,
            "validate": lambda v, n: V.check_positive(v, n),
        },
        "max_seq_len": {
            "default": 512,
            "validate": lambda v, n: V.check_positive(v, n),
        },
        "dropout": {
            "default": 0.1,
            "validate": lambda v, n: V.check_range(v, 0.0, 1.0, n),
        },
        "activation": {
            "default": "swiglu",
            "validate": lambda v, n: V.check_in(v, ["relu", "gelu", "silu", "swiglu"], n),
        },
        "norm_type": {
            "default": "rmsnorm",
            "validate": lambda v, n: V.check_in(v, ["layernorm", "rmsnorm"], n),
        },
        "pre_norm": {
            "default": True,
            "validate": lambda v, n: isinstance(v, bool),
        },
        "tie_embeddings": {
            "default": True,
            "validate": lambda v, n: isinstance(v, bool),
        },
        "vocab_size": {
            "default": 8192,
            "validate": lambda v, n: V.check_positive(v, n),
        },
        # Sharing the input embedding with the output projection removes a
        # vocab_size x d_model matrix and usually improves small models.
        "weight_tying": {"default": True, "validate": lambda v, n: isinstance(v, bool)},
        # Modern decoders drop biases from every projection: they cost
        # parameters and bandwidth for no measurable quality gain.
        "bias": {"default": False, "validate": lambda v, n: isinstance(v, bool)},
        # Grouped-query attention. None (or equal to n_heads) means standard MHA;
        # a smaller value shrinks the KV cache by n_heads / n_kv_heads.
        "n_kv_heads": {
            "default": None,
            "validate": lambda v, n: V.check_positive(v, n) if v is not None else None,
        },
        "position_encoding": {
            "default": "rope",
            "validate": lambda v, n: V.check_in(
                v, ["rope", "alibi", "sinusoidal", "learned", "none"], n
            ),
        },
        "rope_base": {
            "default": 10000.0,
            "validate": lambda v, n: V.check_positive(v, n),
        },
        # >1.0 interpolates positions, extending usable context beyond training.
        "rope_scaling": {
            "default": 1.0,
            "validate": lambda v, n: V.check_positive(v, n),
        },
        # Fused SDPA / FlashAttention kernels instead of the hand-written path.
        "use_flash": {"default": False, "validate": lambda v, n: isinstance(v, bool)},
        # Recompute activations in backward: much less memory, ~30% slower.
        "gradient_checkpointing": {"default": False, "validate": lambda v, n: isinstance(v, bool)},
    }


class OptimizerConfig(Config):
    _schema = {
        "optimizer": {
            "default": "adamw",
            "validate": lambda v, n: V.check_in(v, ["sgd", "adam", "adamw"], n),
        },
        "learning_rate": {
            "default": 3e-4,
            "validate": lambda v, n: v > 0,
        },
        "weight_decay": {
            "default": 0.1,
            "validate": lambda v, n: None,
        },
        "beta1": {
            "default": 0.9,
            "validate": lambda v, n: V.check_range(v, 0.0, 1.0, n),
        },
        "beta2": {
            "default": 0.95,
            "validate": lambda v, n: V.check_range(v, 0.0, 1.0, n),
        },
        "epsilon": {
            "default": 1e-8,
            "validate": lambda v, n: v > 0,
        },
        "max_grad_norm": {
            "default": 1.0,
            "validate": lambda v, n: V.check_positive(v, n) if v is not None else None,
        },
        "gradient_accumulation_steps": {
            "default": 1,
            "validate": lambda v, n: V.check_positive(v, n),
        },
    }


class SchedulerConfig(Config):
    _schema = {
        "scheduler": {
            "default": "cosine",
            "validate": lambda v, n: V.check_in(v, ["constant", "linear", "cosine", "warmup_cosine", "warmup_constant", "warmup_linear"], n),
        },
        "warmup_steps": {
            "default": 1000,
            "validate": lambda v, n: V.check_non_negative(v, n),
        },
        "min_lr_ratio": {
            "default": 0.1,
            "validate": lambda v, n: V.check_range(v, 0.0, 1.0, n),
        },
    }


class DataConfig(Config):
    _schema = {
        "data_paths": {
            "default": [],
            "validate": lambda v, n: isinstance(v, list),
        },
        "batch_size": {
            "default": 32,
            "validate": lambda v, n: V.check_positive(v, n),
        },
        "max_seq_len": {
            "default": 512,
            "validate": lambda v, n: V.check_positive(v, n),
        },
        "shuffle_buffer_size": {
            "default": 10000,
            "validate": lambda v, n: V.check_positive(v, n),
        },
        "num_workers": {
            "default": 4,
            "validate": lambda v, n: V.check_non_negative(v, n),
        },
        "prefetch_factor": {
            "default": 2,
            "validate": lambda v, n: V.check_positive(v, n),
        },
        "validation_split": {
            "default": 0.0,
            "validate": lambda v, n: V.check_range(v, 0.0, 1.0, n),
        },
        "cache_dir": {
            "default": None,
        },
        "bucketing": {
            "default": True,
            "validate": lambda v, n: isinstance(v, bool),
        },
        "pack_sequences": {
            "default": True,
            "validate": lambda v, n: isinstance(v, bool),
        },
    }


class CheckpointConfig(Config):
    _schema = {
        "save_dir": {
            "default": "./checkpoints",
        },
        "save_every_steps": {
            "default": 1000,
            "validate": lambda v, n: V.check_positive(v, n),
        },
        "save_best": {
            "default": True,
            "validate": lambda v, n: isinstance(v, bool),
        },
        "keep_last_n": {
            "default": 5,
            "validate": lambda v, n: V.check_positive(v, n),
        },
        "save_optimizer": {
            "default": True,
            "validate": lambda v, n: isinstance(v, bool),
        },
        "save_scheduler": {
            "default": True,
            "validate": lambda v, n: isinstance(v, bool),
        },
        "resume_from": {
            "default": None,
        },
    }


class LoggingConfig(Config):
    _schema = {
        "log_level": {
            "default": "info",
            "validate": lambda v, n: V.check_in(v, ["trace", "debug", "info", "warning", "error", "fatal"], n),
        },
        "log_file": {
            "default": None,
        },
        "wandb_project": {
            "default": None,
        },
        "log_every_steps": {
            "default": 10,
            "validate": lambda v, n: V.check_positive(v, n),
        },
    }


class HardwareConfig(Config):
    _schema = {
        "device": {
            "default": "auto",
            "validate": lambda v, n: V.check_in(v, ["auto", "cpu", "cuda", "mps"], n),
        },
        # bfloat16 is the safe autocast default: fp32's exponent range, so no
        # loss scaling and no overflow, at half the memory bandwidth.
        "dtype": {
            "default": "bfloat16",
            "validate": lambda v, n: V.check_in(v, ["float16", "float32", "bfloat16", "float64"], n),
        },
        "use_mixed_precision": {
            "default": False,
            "validate": lambda v, n: isinstance(v, bool),
        },
        "num_gpus": {
            "default": 0,
            "validate": lambda v, n: V.check_non_negative(v, n),
        },
        # torch.compile traces and fuses the graph. Large speedup on GPU after a
        # slow first step; off by default so short runs are not dominated by it.
        "compile_model": {"default": False, "validate": lambda v, n: isinstance(v, bool)},
    }


class TrainConfig(Config):
    _schema = {
        "model": {"default": None, "type": ModelConfig},
        "optimizer": {"default": None, "type": OptimizerConfig},
        "scheduler": {"default": None, "type": SchedulerConfig},
        "data": {"default": None, "type": DataConfig},
        "checkpoint": {"default": None, "type": CheckpointConfig},
        "logging": {"default": None, "type": LoggingConfig},
        "hardware": {"default": None, "type": HardwareConfig},
        "tokenizer": {"default": None, "type": TokenizerConfig},
        "seed": {"default": 42, "validate": lambda v, n: isinstance(v, int)},
        "num_epochs": {"default": 1, "validate": lambda v, n: V.check_positive(v, n)},
        "max_steps": {"default": None},
        "eval_every_steps": {"default": 500, "validate": lambda v, n: V.check_positive(v, n)},
        "eval_steps": {"default": 100, "validate": lambda v, n: V.check_positive(v, n)},
        "early_stopping_patience": {"default": None},
        "save_training_state": {"default": True},
    }

    def __init__(self, **kwargs):
        sub_configs = {
            "model": ModelConfig,
            "optimizer": OptimizerConfig,
            "scheduler": SchedulerConfig,
            "data": DataConfig,
            "checkpoint": CheckpointConfig,
            "logging": LoggingConfig,
            "hardware": HardwareConfig,
            "tokenizer": TokenizerConfig,
        }
        for key, cls_type in sub_configs.items():
            if key in kwargs and isinstance(kwargs[key], dict):
                kwargs[key] = cls_type(**kwargs[key])
            elif key not in kwargs:
                kwargs[key] = cls_type()
        super().__init__(**kwargs)

    def resolve_device(self):
        """Pick the device, preferring the GPU with the most free memory.

        Also sizes PyTorch's thread pool to the CPUs this process may actually
        use, which on a container is the cgroup quota rather than the host's
        core count. See myai/core/device.py.
        """
        from ..core.device import setup
        self.hardware.device = str(setup(self.hardware.device))
        return self.hardware.device

    def effective_batch_size(self) -> int:
        """Batch size actually seen per optimizer step."""
        return self.data.batch_size * self.optimizer.gradient_accumulation_steps


# Ready-made model scales. Each keeps head_dim at 64, which is what attention
# kernels are tuned for, and d_ff at ~4x d_model.
MODEL_PRESETS = {
    "tiny": dict(d_model=128, n_heads=4, n_kv_heads=2, n_layers=4, d_ff=512, max_seq_len=256),
    "mini": dict(d_model=256, n_heads=4, n_kv_heads=2, n_layers=6, d_ff=1024, max_seq_len=512),
    "small": dict(d_model=512, n_heads=8, n_kv_heads=4, n_layers=8, d_ff=2048, max_seq_len=1024),
    "base": dict(d_model=768, n_heads=12, n_kv_heads=4, n_layers=12, d_ff=3072, max_seq_len=1024),
    # ~1B parameters at vocab 8192. The architecture scales to this cleanly, but
    # the hardware requirement is a step change, not a longer wait:
    #   weights 3.9 GB + grads 3.9 GB + Adam m/v 7.9 GB = 15.7 GB of fp32 state
    #   before a single activation, so it needs a 40 GB+ accelerator.
    # See docs/SCALING.md for the measured basis of those numbers.
    "xl": dict(d_model=1536, n_heads=12, n_kv_heads=4, n_layers=28, d_ff=6144, max_seq_len=1024),
}


def preset_config(name: str, **overrides) -> "TrainConfig":
    """Build a TrainConfig from a named model scale.

    >>> config = preset_config("small", max_steps=1000)
    """
    if name not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {sorted(MODEL_PRESETS)}")

    model_kwargs = dict(MODEL_PRESETS[name])
    model_kwargs.update(overrides.pop("model", {}))

    config = TrainConfig(model=model_kwargs, **overrides)
    config.data.max_seq_len = config.model.max_seq_len
    return config
