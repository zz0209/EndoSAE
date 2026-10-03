from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.acknowledgement_sae import AcknowledgementPredictor, supervised_contrastive_loss
from src.checkpoint_io import read_json


METHODS = ("sparse_shared", "dense_shared", "sparse_self", "dense_self")


class TemporalSharedSAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.method = config["method"]
        self.input_dim = int(config["input_dim"])
        self.shared_dim = int(config["shared_dim"])
        self.private_dim = int(config["private_dim"])
        self.latent_dim = int(config["readout_dim"])
        self.top_k = int(config["top_k"])
        if self.method not in METHODS or not 0 < self.top_k <= min(self.shared_dim, self.private_dim):
            raise ValueError("Invalid temporal component configuration")
        self.encoder = nn.Linear(self.input_dim, self.shared_dim + self.private_dim)
        self.decoder = nn.Linear(self.shared_dim + self.private_dim, self.input_dim)
        self.readout = nn.Linear(self.shared_dim, self.latent_dim, bias=False)
        with torch.no_grad():
            nn.init.normal_(self.decoder.weight, std=self.input_dim ** -0.5)
            self.normalize_decoder()
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()
            self.decoder.bias.zero_()

    @torch.no_grad()
    def normalize_decoder(self):
        norms = self.decoder.weight.norm(dim=0, keepdim=True)
        if not torch.isfinite(norms).all() or torch.any(norms <= 0):
            raise ValueError("Invalid decoder direction")
        self.decoder.weight.div_(norms)

    def encode_parts(self, features):
        shared, private = F.relu(self.encoder(features)).split([self.shared_dim, self.private_dim], dim=-1)
        if self.method.startswith("sparse_"):
            parts = []
            for part in (shared, private):
                values, indices = torch.topk(part, self.top_k, dim=-1, sorted=False)
                parts.append(torch.zeros_like(part).scatter(-1, indices, values))
            shared, private = parts
        return shared, private

    def reconstruct(self, shared, private):
        return self.decoder(torch.cat([shared, private], dim=-1))

    def encode(self, features):
        shared, _ = self.encode_parts(features)
        return self.readout(shared)

    @staticmethod
    def memory(codes):
        return F.normalize(codes, dim=-1)

    def pair_scores(self, source, query):
        return (self.memory(source) * self.memory(query)).sum(-1)


def training_loss(model, inputs, labels, batch, config):
    shared, private = model.encode_parts(inputs)
    projected = model.readout(shared)
    unit = F.normalize(projected, dim=-1)
    source, positive, negative = [batch[key] for key in ("source", "positive", "negative")]
    positive_score = (unit[source] * unit[positive]).sum(-1)
    negative_score = (unit[source] * unit[negative]).sum(-1)
    losses = {
        "reconstruction": F.mse_loss(model.reconstruct(shared, private), inputs),
        "identity": supervised_contrastive_loss(projected, labels, float(config["temperature"])),
        "episode": F.softplus((negative_score - positive_score + float(config["episode_margin"]))
                              / float(config["temperature"])).mean(),
        "swap": .5 * (F.mse_loss(model.reconstruct(shared[positive], private[source]), inputs[source])
                      + F.mse_loss(model.reconstruct(shared[source], private[positive]), inputs[positive])),
        "consistency": (F.normalize(shared[source], dim=-1) - F.normalize(shared[positive], dim=-1))
                       .square().sum(-1).mean(),
    }
    total = sum(float(config[key + "_weight"]) * losses[key] for key in ("reconstruction", "identity", "episode"))
    if model.method.endswith("_shared"):
        total = total + sum(float(config[key + "_weight"]) * losses[key] for key in ("swap", "consistency"))
    return total, losses, shared, private


def load_predictor(directory, device="cpu"):
    directory = Path(directory)
    model = TemporalSharedSAE(read_json(directory / "model_config.json"))
    with np.load(directory / "model.npz", allow_pickle=False) as saved:
        model.load_state_dict({key: torch.from_numpy(saved[key].copy()) for key in saved.files}, strict=True)
    with np.load(directory / "normalization.npz", allow_pickle=False) as saved:
        mean, scale = saved["mean"].copy(), saved["scale"].copy()
    return AcknowledgementPredictor(model, mean, scale, device)
