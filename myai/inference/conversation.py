"""Interactive conversation handler for chat-based interactions."""

from typing import Optional, List, Dict, Any, Callable
from ..core.logging import logger
from ..core.errors import InferenceError
from .generation import AutoregressiveGenerator


class ConversationHandler:
    """Manages interactive conversations with the language model.

    Maintains conversation history, formats prompts for chat,
    and handles multi-turn interactions.
    """

    def __init__(
        self,
        generator: AutoregressiveGenerator,
        system_prompt: str = "You are a helpful AI assistant.",
        max_history: int = 10,
    ):
        self._generator = generator
        self._system_prompt = system_prompt
        self._max_history = max_history
        self._history: List[Dict[str, str]] = []

    def add_user_message(self, message: str) -> None:
        self._history.append({"role": "user", "content": message})
        if len(self._history) > self._max_history * 2:
            self._history = self._history[-self._max_history * 2:]

    def add_assistant_message(self, message: str) -> None:
        self._history.append({"role": "assistant", "content": message})

    def _format_prompt(self) -> str:
        """Format the conversation history into a prompt."""
        parts = [f"System: {self._system_prompt}"]

        for msg in self._history:
            role = msg["role"].capitalize()
            parts.append(f"{role}: {msg['content']}")

        parts.append("Assistant:")
        return "\n".join(parts)

    def chat(
        self,
        message: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
    ) -> str:
        """Send a message and get a response."""
        self.add_user_message(message)

        prompt = self._format_prompt()
        # return_full_text=False keeps the prompt out of the reply - otherwise
        # the whole transcript gets appended to the history every turn.
        response = self._generator.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            return_full_text=False,
            stop_sequences=["\nUser:", "\nSystem:", "\nAssistant:"],
        ).strip()

        self.add_assistant_message(response)
        return response

    def reset(self) -> None:
        """Clear conversation history."""
        self._history = []

    @property
    def history(self) -> List[Dict[str, str]]:
        return list(self._history)
