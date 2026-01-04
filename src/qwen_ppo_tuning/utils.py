import torch
import torch.nn.functional as F


def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=-1)
    logpy = torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return logpy


def pad_sequences(seqs, pad_value, padding="right", pad_to: int = None):  # type: ignore
    """
    Padding sequence to the same length
    """
    max_len = max(len(seq) for seq in seqs) if pad_to is None else pad_to
    if padding == "right":
        padded_seqs = [seq + [pad_value] * (max_len - len(seq)) for seq in seqs]
    elif padding == "left":
        padded_seqs = [[pad_value] * (max_len - len(seq)) + seq for seq in seqs]
    else:
        assert ValueError
    return padded_seqs


def calculate_clip_fraction(ratio, clip_range):
    return ((ratio > 1 + clip_range) | (ratio < 1 - clip_range)).float().mean().item()


def calculate_value_clip_fraction(values, old_values, clip_range):
    values_clipped = torch.clamp(values, old_values - clip_range, old_values + clip_range)
    return (values_clipped != values).float().mean().item()


def calculate_entropy(logits, mask):
    """
    Calculate entropy from logits with proper masking.

    Args:
        logits: Raw logits from the model (batch_size, seq_len, vocab_size)
        mask: Mask indicating which tokens to include in entropy calculation

    Returns:
        Mean entropy across valid tokens
    """

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    # Calculate log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)
    # Calculate entropy: -sum(p * log(p))
    entropy = -torch.sum(probs * log_probs, dim=-1)  # (batch_size, seq_len)
    # Apply mask and calculate mean
    masked_entropy = entropy * mask
    n_valid_tokens = mask.sum()

    if n_valid_tokens > 0:
        mean_entropy = masked_entropy.sum() / n_valid_tokens
    else:
        mean_entropy = torch.tensor(0.0, device=entropy.device)

    return mean_entropy


class RunningMoments:
    def __init__(self):
        self.mean = 0
        self.var = 1
        self.std = 1
        self.count = 1e-24

    @torch.no_grad
    def update(self, xs):
        xs_count = xs.numel()
        xs_var, xs_mean = torch.var_mean(xs, unbiased=False)
        xs_mean, xs_var = xs_mean.float(), xs_var.float()
        delta = xs_mean - self.mean
        tot_count = self.count + xs_count

        new_sum = xs_var * xs_count
        # correct old_sum deviation accounting for the new mean
        old_sum = self.var * self.count + delta**2 * self.count * xs_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * xs_count / tot_count
        self.var = tot_sum / tot_count
        self.std = (self.var * tot_count / (tot_count - 1)).float().sqrt()
        self.count = tot_count

        return xs_mean.item(), (xs_var * xs_count / (xs_count - 1)).float().sqrt().item()


@torch.no_grad()
def whiten(xs: torch.Tensor, mask: torch.BoolTensor, shift_mean=True) -> torch.Tensor:
    """
    Whitens values
    """

    mean = xs.sum() / mask.sum()
    var = torch.sum(((xs - mean) ** 2).mul(mask)) / mask.sum()

    whitened = (xs - mean) * torch.rsqrt(var + 1e-6)
    if not shift_mean:
        whitened += mean
    return whitened
