"""Loss functions: weighted BCE, function BCE, combined, CTL loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_bce_loss(boundary_logits, boundary_target, pos_weight=10.0):
    """Weighted BCE for sparse boundary targets.

    Args:
        boundary_logits: (B, T', 1)
        boundary_target: (B, T', 1)
        pos_weight: weight for positive (boundary) frames

    Returns:
        scalar loss
    """
    loss = F.binary_cross_entropy_with_logits(
        boundary_logits, boundary_target,
        pos_weight=torch.tensor([pos_weight], device=boundary_logits.device),
    )
    return loss


def function_bce_loss(function_logits, function_target):
    """Multi-label BCE for 7 function classes.

    Args:
        function_logits: (B, T', 7)
        function_target: (B, T', 7)

    Returns:
        scalar loss
    """
    loss = F.binary_cross_entropy_with_logits(
        function_logits, function_target,
        reduction="mean",
    )
    return loss


def combined_loss(boundary_logits, boundary_target, function_logits, function_target,
                  boundary_weight=0.9, function_weight=0.1):
    """Weighted sum of boundary and function losses (Variant A)."""
    b_loss = weighted_bce_loss(boundary_logits, boundary_target)
    f_loss = function_bce_loss(function_logits, function_target)
    return boundary_weight * b_loss + function_weight * f_loss


class CTLLoss(nn.Module):
    """CTC-based loss over class token sequences (Variant B).

    CTL encourages frame-level predictions to be consistent with
    the ordered sequence of section labels.
    """

    def __init__(self, blank=7, zero_infinity=True):
        super().__init__()
        self.blank = blank
        self.ctc = nn.CTCLoss(blank=blank, zero_infinity=zero_infinity)

    def forward(self, function_logits, token_sequences, input_lengths, target_lengths):
        """Compute CTL loss.

        Args:
            function_logits: (B, T', 7) — logits for 7 function classes
            token_sequences: list of list[int] or padded (B, max_tokens) — target tokens
            input_lengths: (B,) — number of valid frames per sample
            target_lengths: (B,) — number of tokens per target sequence

        Returns:
            scalar loss
        """
        B, T, C = function_logits.shape
        blank_logits = torch.zeros(B, T, 1, device=function_logits.device)
        logits = torch.cat([function_logits, blank_logits], dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.permute(1, 0, 2)

        if isinstance(token_sequences, list):
            max_tokens = max(len(s) for s in token_sequences)
            padded = torch.full((B, max_tokens), self.blank, dtype=torch.long, device=function_logits.device)
            for i, seq in enumerate(token_sequences):
                padded[i, :len(seq)] = torch.as_tensor(seq, dtype=torch.long, device=function_logits.device)
                # padded[i, :len(seq)] = seq.to(device=function_logits.device, dtype=torch.long)
            token_sequences = padded

        loss = self.ctc(log_probs, token_sequences, input_lengths, target_lengths)
        return loss


class CombinedWithCTLLoss(nn.Module):
    """Combined loss with CTL (Variant B)."""

    def __init__(self, boundary_weight=0.9, function_weight=0.1, ctl_weight=0.1, blank=7):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.function_weight = function_weight
        self.ctl_weight = ctl_weight
        self.ctl = CTLLoss(blank=blank)

    def forward(self, boundary_logits, boundary_target, function_logits, function_target,
                token_sequences, input_lengths, target_lengths):
        b_loss = weighted_bce_loss(boundary_logits, boundary_target)
        f_loss = function_bce_loss(function_logits, function_target)
        ctl_loss = self.ctl(function_logits, token_sequences, input_lengths, target_lengths)
        return (self.boundary_weight * b_loss +
                self.function_weight * f_loss +
                self.ctl_weight * ctl_loss), ctl_loss
