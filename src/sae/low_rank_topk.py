"""Low-rank encoder/decoder updates around a frozen project TopK dictionary."""
import math
import torch


class LowRankTopK(torch.nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = rank
        self.initial_base_state = {key: value.detach().clone() for key, value in base.state_dict().items()}
        width, dimension = base.encoder.weight.shape
        device = base.encoder.weight.device
        self.encoder_a = torch.nn.Parameter(torch.empty(rank, dimension, device=device))
        self.encoder_b = torch.nn.Parameter(torch.zeros(width, rank, device=device))
        self.decoder_a = torch.nn.Parameter(torch.empty(rank, width, device=device))
        self.decoder_b = torch.nn.Parameter(torch.zeros(dimension, rank, device=device))
        torch.nn.init.kaiming_uniform_(self.encoder_a, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.decoder_a, a=math.sqrt(5))

    def encode_inference(self, values):
        centered = values - self.base.b_dec
        pre = torch.relu(self.base.encoder(centered) + (centered @ self.encoder_a.T) @ self.encoder_b.T)
        selected = pre.topk(int(self.base.k.item()), dim=-1, sorted=False)
        return torch.zeros_like(pre).scatter(1, selected.indices, selected.values)

    def effective_decoder(self):
        weight = self.base.decoder.weight + self.decoder_b @ self.decoder_a
        return weight / weight.norm(dim=0, keepdim=True).clamp_min(1e-12)

    def decode(self, codes):
        return torch.nn.functional.linear(codes, self.effective_decoder()) + self.base.b_dec

    def base_unchanged(self):
        return all(torch.equal(value, self.initial_base_state[key]) for key, value in self.base.state_dict().items())

    def materialized_state(self):
        state = {key: value.detach().clone() for key, value in self.base.state_dict().items()}
        state['encoder.weight'] = (self.base.encoder.weight + self.encoder_b @ self.encoder_a).detach()
        state['decoder.weight'] = self.effective_decoder().detach()
        return state
