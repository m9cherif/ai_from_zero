"""Error hierarchy for the entire project.

Every subsystem defines its own error type inheriting from MyAIError.
"""


class MyAIError(Exception):
    """Base error for all myai exceptions."""
    pass


class ConfigError(MyAIError):
    """Invalid configuration or configuration mismatch."""
    pass


class TokenizerError(MyAIError):
    """Tokenizer-related errors: encoding, decoding, vocabulary, training."""
    pass


class DataError(MyAIError):
    """Data pipeline errors: ingestion, cleaning, filtering, streaming."""
    pass


class NNError(MyAIError):
    """Neural network errors: shape mismatch, initialization, forward/backward."""
    pass


class TrainingError(MyAIError):
    """Training engine errors: optimizer, scheduler, loop, stability."""
    pass


class InferenceError(MyAIError):
    """Inference engine errors: generation, sampling, conversation."""
    pass


class CheckpointError(MyAIError):
    """Checkpoint save/load errors: missing files, corruption, version mismatch."""
    pass
