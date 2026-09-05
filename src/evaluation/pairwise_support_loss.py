"""Within-frame pairwise logistic ranking; RankNet-style loss, not a new loss."""
import torch


def pairwise_support_loss(logits, labels, token_indices):
    """Pairs are adjacent positive/negative patches from the same 196-token frame."""
    if logits.ndim != 1 or len(logits) % 2 or logits.shape != labels.shape:
        raise ValueError('Expected equally sized even 1-D logits and labels')
    if token_indices.shape != logits.shape:
        raise ValueError('Token indices must match logits')
    if not bool(((labels[0::2] == 1) & (labels[1::2] == 0)).all()):
        raise ValueError('Pairs must be ordered inside then outside')
    if not bool((token_indices[0::2] // 196 == token_indices[1::2] // 196).all()):
        raise ValueError('Ranking pairs must share a frame')
    return torch.nn.functional.softplus(-(logits[0::2] - logits[1::2])).mean()
