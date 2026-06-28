import torch
import torch.nn.functional as F

from src.data_utils import NUM_FUNCTIONS, FUNCTION_LABELS


def ctl_loss(
    logits: torch.Tensor,
    target_tokens: list[torch.Tensor],
    blank_logprob: float = -1e10,
) -> torch.Tensor:
    """
    Connectionist Temporal Localization (CTL) loss.
    Computes -log P(target_sequence | model_logits) using forward-backward algorithm.

    Unlike CTC, CTL has no blank token. Each frame is assigned to exactly one
    target token, and tokens appear in order. Consecutive frames with the same
    label correspond to the same token continuing.

    Args:
        logits: (B, T, C) raw logits before softmax
        target_tokens: list of B tensors, each of shape (S_i,) with token indices
        blank_logprob: placeholder for -inf in log space

    Returns:
        scalar CTL loss averaged over batch
    """
    B, T, C = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)  # (B, T, C)

    total_loss = 0.0
    for b in range(B):
        tokens = target_tokens[b]  # (S,)
        S = tokens.shape[0]
        if S == 0 or S > T:
            continue

        log_alpha = torch.full((T + 1, S), blank_logprob, device=logits.device)
        log_alpha[0, 0] = 0.0

        for t in range(1, T + 1):
            log_yt = log_probs[b, t - 1, :]

            max_s = min(t, S)
            for s in range(max_s):
                token_idx = tokens[s]
                log_emit = log_yt[token_idx]

                stay = log_alpha[t - 1, s]
                trans = log_alpha[t - 1, s - 1] if s > 0 else blank_logprob
                log_alpha[t, s] = log_emit + torch.logsumexp(
                    torch.tensor([stay, trans]), dim=0
                )

        log_likelihood = log_alpha[T, S - 1]
        total_loss = total_loss - log_likelihood

    return total_loss / B


def ctl_loss_batch(
    logits: torch.Tensor,
    target_padded: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_logprob: float = -1e10,
) -> torch.Tensor:
    """
    Batched CTL loss using padded targets.

    Args:
        logits: (B, T, C) raw logits
        target_padded: (B, S_max) padded token sequences
        target_lengths: (B,) length of each target sequence

    Returns:
        scalar CTL loss
    """
    B, T, C = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)

    total_loss = 0.0
    for b in range(B):
        S = target_lengths[b].item()
        if S == 0 or S > T:
            continue
        tokens = target_padded[b, :S]
        log_alpha = torch.full((T + 1, S), blank_logprob, device=logits.device)
        log_alpha[0, 0] = 0.0

        for t in range(1, T + 1):
            log_yt = log_probs[b, t - 1, :]
            max_s = min(t, S)
            for s in range(max_s):
                log_emit = log_yt[tokens[s]]
                stay = log_alpha[t - 1, s]
                trans = log_alpha[t - 1, s - 1] if s > 0 else blank_logprob
                log_alpha[t, s] = log_emit + torch.logsumexp(
                    torch.tensor([stay, trans]), dim=0
                )

        log_likelihood = log_alpha[T, S - 1]
        total_loss = total_loss - log_likelihood

    return total_loss / B
