import torch
import torch.nn.functional as F


def ctl_loss_batch(
    logits: torch.Tensor,
    target_padded: torch.Tensor,
    target_lengths: torch.Tensor,
    blank_logprob: float = -1e10,
) -> torch.Tensor:
    """
    Batched CTL loss using vectorized forward-backward.

    Args:
        logits: (B, T, C) raw logits
        target_padded: (B, S_max) padded token sequences
        target_lengths: (B,) length of each target sequence

    Returns:
        scalar CTL loss averaged over valid samples
    """
    B, T, C = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)
    S_max = target_padded.shape[1]

    if S_max == 0:
        return torch.tensor(0.0, device=logits.device)

    token_log_probs = log_probs.gather(
        2, target_padded.unsqueeze(1).expand(-1, T, -1)
    )

    mask_S = torch.arange(S_max, device=logits.device).unsqueeze(0) >= target_lengths.unsqueeze(1)
    mask_S = mask_S.unsqueeze(1)
    token_log_probs = torch.where(
        mask_S.expand(-1, T, -1),
        torch.tensor(blank_logprob, device=logits.device),
        token_log_probs,
    )

    log_alpha = torch.full((B, T + 1, S_max), blank_logprob, device=logits.device)
    log_alpha = list(log_alpha.unbind(1))
    log_alpha[0] = log_alpha[0].clone()
    log_alpha[0][:, 0] = 0.0

    alpha_prev = log_alpha[0]
    for t in range(1, T + 1):
        stay = alpha_prev
        trans = torch.full_like(stay, blank_logprob)
        trans[:, 1:] = alpha_prev[:, :-1]

        cur = token_log_probs[:, t - 1, :] + torch.logaddexp(stay, trans)

        mask_s_gt_t = torch.arange(S_max, device=logits.device).unsqueeze(0) >= t
        cur = torch.where(
            mask_s_gt_t | mask_S.squeeze(1),
            torch.tensor(blank_logprob, device=logits.device),
            cur,
        )
        log_alpha[t] = cur
        alpha_prev = cur

    log_alpha = torch.stack(log_alpha, dim=1)

    batch_idx = torch.arange(B, device=logits.device)
    S_last = (target_lengths - 1).clamp(min=0, max=S_max - 1)
    log_likelihood = log_alpha[batch_idx, T, S_last]

    valid = (target_lengths > 0) & (target_lengths <= T)
    if not valid.any():
        return torch.tensor(0.0, device=logits.device)

    loss = -log_likelihood[valid].mean()
    return loss
