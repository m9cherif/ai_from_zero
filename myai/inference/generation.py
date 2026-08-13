"""Autoregressive text generation."""

from typing import Optional, List, Callable, Iterator, Dict, Any
import torch
from ..core.types import Tensor
from ..core.logging import logger
from ..core.errors import InferenceError
from ..nn.model import LanguageModel
from ..nn.logits import apply_repetition_penalty, sample_from_logits
from .sampling import Sampler, TemperatureSampler, TopKSampler, TopPSampler, GreedySampler


class AutoregressiveGenerator:
    """Autoregressive text generator.

    Generates text token by token, feeding previously generated tokens back as
    input. Uses the model's KV cache, so cost grows linearly with the number of
    generated tokens rather than quadratically.

    ``generate`` returns a string. Streaming is a separate method
    (``generate_stream``) rather than a flag: a function containing ``yield``
    is a generator function in Python, so a single method would return a
    generator object instead of text for *every* caller, streaming or not.
    """

    def __init__(
        self,
        model: LanguageModel,
        tokenizer=None,
        max_length: int = 2048,
        device: Optional[torch.device] = None,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._device = device or model.device

    def _build_sampler(
        self,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        greedy: bool = False,
    ) -> Sampler:
        """Build a sampler from generation parameters."""
        if greedy:
            return GreedySampler()
        if top_p is not None:
            return TopPSampler(p=top_p, temperature=temperature)
        if top_k is not None:
            return TopKSampler(k=top_k, temperature=temperature)
        return TemperatureSampler(temperature=temperature)

    def _require_tokenizer(self):
        if self._tokenizer is None:
            raise InferenceError("Tokenizer required for text generation")

    def _encode(self, prompt: str) -> Tensor:
        ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            bos = getattr(self._tokenizer, "bos_token_id", None)
            ids = [bos if bos is not None else 0]
        return torch.tensor([ids], dtype=torch.long, device=self._device)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        return_full_text: bool = True,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature; <= 0 means greedy
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            min_p: Min-p sampling parameter
            eos_token_id: End-of-sequence token ID
            repetition_penalty: >1.0 discourages repeats
            stop_sequences: Strings that end generation when produced
            return_full_text: Include the prompt in the returned string

        Returns:
            The generated text
        """
        self._require_tokenizer()

        if eos_token_id is None:
            eos_token_id = getattr(self._tokenizer, "eos_token_id", None)

        input_tensor = self._encode(prompt)
        prompt_len = input_tensor.shape[1]

        output = self._model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
        )

        ids = output[0].tolist()
        if not return_full_text:
            ids = ids[prompt_len:]

        text = self._tokenizer.decode(ids, skip_special_tokens=True)
        return self._apply_stop_sequences(text, stop_sequences, prompt if return_full_text else "")

    @staticmethod
    def _apply_stop_sequences(
        text: str, stop_sequences: Optional[List[str]], protected_prefix: str
    ) -> str:
        """Truncate at the earliest stop sequence found after the prompt."""
        if not stop_sequences:
            return text

        search_from = len(protected_prefix)
        cut = len(text)
        for stop in stop_sequences:
            idx = text.find(stop, search_from)
            if idx != -1:
                cut = min(cut, idx)
        return text[:cut]

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
    ) -> Iterator[str]:
        """Yield decoded text incrementally as each token is produced."""
        self._require_tokenizer()

        if eos_token_id is None:
            eos_token_id = getattr(self._tokenizer, "eos_token_id", None)

        model = self._model
        was_training = model.training
        model.eval()

        generated = self._encode(prompt)
        max_seq_len = model.config.max_seq_len

        cache = model.build_cache(
            batch_size=1,
            max_seq_len=min(max_seq_len, generated.shape[1] + max_new_tokens),
        )

        step_input = generated
        offset = 0
        emitted = ""

        try:
            for _ in range(max_new_tokens):
                outputs = model(step_input, cache=cache, offset=offset, num_logits_to_keep=1)
                logits = outputs["logits"][:, -1, :].float()
                logits = apply_repetition_penalty(logits, generated, repetition_penalty)

                next_token = sample_from_logits(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p, min_p=min_p
                )
                token_id = int(next_token.item())
                generated = torch.cat([generated, next_token], dim=1)

                if eos_token_id is not None and token_id == eos_token_id:
                    break

                # Decode the whole continuation each step and emit only the new
                # suffix: multi-byte and merged tokens do not decode correctly
                # in isolation.
                full = self._tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
                chunk = full[len(emitted):]
                if chunk:
                    emitted = full
                    yield chunk

                if stop_sequences and any(s in emitted for s in stop_sequences):
                    break

                offset = cache.length
                step_input = next_token
                if offset >= max_seq_len:
                    break
        finally:
            if was_training:
                model.train()

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.0,
    ) -> List[str]:
        """Generate for multiple prompts in a single batched forward pass.

        Prompts are left-padded so every sequence ends at the same position,
        which is what lets one KV cache serve the whole batch.
        """
        self._require_tokenizer()

        if eos_token_id is None:
            eos_token_id = getattr(self._tokenizer, "eos_token_id", None)
        pad_id = getattr(self._tokenizer, "pad_token_id", 0) or 0

        encoded = [self._tokenizer.encode(p, add_special_tokens=False) or [pad_id] for p in prompts]
        max_len = max(len(ids) for ids in encoded)
        padded = [[pad_id] * (max_len - len(ids)) + ids for ids in encoded]

        input_tensor = torch.tensor(padded, dtype=torch.long, device=self._device)

        output = self._model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            eos_token_id=eos_token_id,
            pad_token_id=pad_id,
            repetition_penalty=repetition_penalty,
        )

        results = []
        for row, original in zip(output, encoded):
            ids = [t for t in row.tolist() if t != pad_id]
            results.append(self._tokenizer.decode(ids, skip_special_tokens=True))
        return results
