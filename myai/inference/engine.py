"""Inference engine for loading checkpoints and running generation."""

from typing import Optional, List, Dict, Any, Iterator
from pathlib import Path
import torch
from ..core.logging import logger
from ..core.errors import InferenceError
from ..nn.model import LanguageModel, LMConfig
from .generation import AutoregressiveGenerator


class InferenceEngine:
    """Standalone inference engine for trained models.

    Operates independently of the training engine. Loads checkpoints
    efficiently and provides generation capabilities.
    """

    def __init__(self, device: Optional[str] = None, dtype: Optional[torch.dtype] = None):
        from ..core.device import setup
        # None/"auto" picks the emptiest GPU and sizes CPU threads to the
        # container's real allowance; an explicit device is honoured as given.
        self._device = setup(device or "auto")
        self._dtype = dtype
        self._model: Optional[LanguageModel] = None
        self._tokenizer = None
        self._generator: Optional[AutoregressiveGenerator] = None

    @staticmethod
    def _extract_model_config(checkpoint: Dict[str, Any]) -> LMConfig:
        """Recover the model geometry from any of the layouts we write."""
        config_data = checkpoint.get("config") or {}
        model_data = config_data.get("model", config_data) if isinstance(config_data, dict) else {}

        if not isinstance(model_data, dict):
            model_data = {}

        merged = dict(model_data)
        # TrainConfig calls it weight_tying; LMConfig calls it tie_embeddings.
        if "weight_tying" in merged and "tie_embeddings" not in merged:
            merged["tie_embeddings"] = merged["weight_tying"]
        if "vocab_size" not in merged and "vocab_size" in checkpoint:
            merged["vocab_size"] = checkpoint["vocab_size"]
        merged.setdefault("padding_idx", 0)

        return LMConfig.from_dict(merged)

    def load_checkpoint(self, checkpoint_path: str, tokenizer_path: Optional[str] = None) -> None:
        """Load a model from a checkpoint file."""
        # A URL or hf:// spec is downloaded and cached; a plain path is used
        # as-is. This is what lets every entry point share one trained model
        # rather than a file that exists on a single machine.
        from ..checkpoint.remote import resolve_checkpoint
        checkpoint_path = resolve_checkpoint(checkpoint_path)

        logger.info(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)

        model_config = self._extract_model_config(checkpoint)
        # Dropout is a training-time regularizer; it must be inert at inference.
        model_config.dropout = 0.0

        self._model = LanguageModel(model_config).to(self._device, dtype=self._dtype)
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.eval()

        logger.info(f"Model loaded: {self._model}")

        if tokenizer_path:
            self._tokenizer = self._load_tokenizer(tokenizer_path)
        elif "tokenizer_state" in checkpoint:
            from ..tokenizer import BPETokenizer
            self._tokenizer = BPETokenizer.from_dict(checkpoint["tokenizer_state"])

        self._generator = AutoregressiveGenerator(
            model=self._model,
            tokenizer=self._tokenizer,
            device=self._device,
        )

    @staticmethod
    def _load_tokenizer(path: str):
        """Load whichever tokenizer flavour the file contains."""
        from ..tokenizer import BPETokenizer, CharTokenizer, load_tokenizer
        return load_tokenizer(path)

    def load_model(self, model: LanguageModel, tokenizer=None) -> None:
        """Attach an already-constructed model (skips disk entirely)."""
        self._model = model.to(self._device)
        self._model.eval()
        self._tokenizer = tokenizer
        self._generator = AutoregressiveGenerator(
            model=self._model, tokenizer=tokenizer, device=self._device
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        return_full_text: bool = True,
    ) -> str:
        """Generate text from a prompt. Returns a string."""
        if self._generator is None:
            raise InferenceError("No model loaded. Call load_checkpoint() first.")
        return self._generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences,
            return_full_text=return_full_text,
        )

    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]:
        """Stream generated text chunk by chunk."""
        if self._generator is None:
            raise InferenceError("No model loaded. Call load_checkpoint() first.")
        return self._generator.generate_stream(prompt=prompt, **kwargs)

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate for several prompts in one batched pass."""
        if self._generator is None:
            raise InferenceError("No model loaded. Call load_checkpoint() first.")
        return self._generator.generate_batch(prompts=prompts, **kwargs)

    @property
    def model(self) -> Optional[LanguageModel]:
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        return self._device
