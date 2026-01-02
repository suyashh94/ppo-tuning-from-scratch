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
