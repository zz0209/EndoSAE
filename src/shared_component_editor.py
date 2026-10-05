import torch
from torch import nn
from torch.nn import functional as F


class TaskDictionary(nn.Module):
    def __init__(self, dimension, width, top_k, sparse):
        super().__init__()
        self.sparse = bool(sparse)
        self.top_k = int(top_k)
        self.encoder = nn.Linear(dimension, width)
        self.decoder = nn.Linear(width, dimension, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.decoder.weight, std=dimension ** -.5)
            self.normalize_decoder()
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()

    @torch.no_grad()
    def normalize_decoder(self):
        lengths = self.decoder.weight.norm(dim=0, keepdim=True)
        if not torch.isfinite(lengths).all() or torch.any(lengths <= 0):
            raise ValueError("Invalid decoder direction")
        self.decoder.weight.div_(lengths)

    def encode(self, value):
        code = F.relu(self.encoder(value))
        if self.sparse:
            values, indices = torch.topk(code, self.top_k, dim=-1, sorted=False)
            code = torch.zeros_like(code).scatter(-1, indices, values)
        return code

    def forward(self, value):
        return self.decoder(self.encode(value))


def shared_coefficients(first, second):
    return torch.where(first * second > 0,
                       first.sign() * torch.minimum(first.abs(), second.abs()),
                       torch.zeros_like(first))


def edit_scores(first, second, first_code, second_code, directions, gains, budget,
                mode="bilateral", rotation=None):
    if mode not in ("bilateral", "source_only"):
        raise ValueError("Unknown component edit mode")
    coefficients = (shared_coefficients(first_code, second_code)
                    if mode == "bilateral" else first_code)
    delta = (coefficients * gains) @ directions.T
    lengths = delta.norm(dim=-1, keepdim=True)
    delta = delta * torch.clamp(budget / lengths.clamp_min(1e-12), max=1.)
    if rotation is not None:
        delta = delta @ rotation
    altered_first = first - delta
    altered_second = second - delta if mode == "bilateral" else second
    scores = (F.normalize(altered_first, dim=-1) * F.normalize(altered_second, dim=-1)).sum(-1)
    return scores, delta, coefficients


def correction_loss(scores, labels, weights, threshold, temperature, gains, penalty):
    signed = torch.where(labels, threshold - scores, scores - threshold) / temperature
    return (weights * F.softplus(signed)).sum() + penalty * gains.mean()
