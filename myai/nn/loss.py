"""Loss functions implemented from scratch.

Uses PyTorch tensor operations but implements the mathematical
computations manually.
"""

import torch
from .module import Module
from ..core.types import Tensor


class CrossEntropyLoss(Module):
    """Cross-entropy loss for language modeling.

    Computes the negative log-likelihood loss between predicted
    logits and target token IDs, ignoring positions with label -100.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Unknown reduction: {reduction}")
        self._reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: Raw logits of shape (batch, seq_len, vocab_size)
            targets: Target token IDs of shape (batch, seq_len)
                     Positions with value -100 are ignored.

        Returns:
            Scalar loss value
        """
        batch_size, seq_len, vocab_size = logits.shape

        # Reshape logits to (batch * seq_len, vocab_size)
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = targets.reshape(-1)

        # Compute log-softmax manually:
        # log_softmax(x_i) = x_i - log(sum(exp(x_j)))
        max_logits = logits_flat.max(dim=-1, keepdim=True).values
        shifted = logits_flat - max_logits
        exp_shifted = torch.exp(shifted)
        sum_exp = exp_shifted.sum(dim=-1, keepdim=True)
        log_probs = shifted - torch.log(sum_exp)

        # Mask invalid targets (-100) to 0 for gather, then mask NLL
        ignore_mask = targets_flat != -100
        safe_targets = targets_flat.clamp(min=0)
        nll = -log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
        nll = nll * ignore_mask.float()

        # Compute loss
        num_valid = ignore_mask.sum().clamp(min=1)

        if self._reduction == "mean":
            return nll.sum() / num_valid
        elif self._reduction == "sum":
            return nll.sum()
        else:  # "none"
            return nll.reshape(batch_size, seq_len)
