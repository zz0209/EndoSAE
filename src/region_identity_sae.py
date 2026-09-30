from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.acknowledgement_sae import AcknowledgementPredictor
from src.checkpoint_io import read_json


METHODS = ("sparse_self", "sparse_canonical", "dense_self", "dense_canonical")


class RegionIdentitySAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.method = config["method"]
        self.input_dim = int(config["input_dim"])
        self.latent_dim = int(config["latent_dim"])
        self.top_k = int(config["top_k"])
        if self.method not in METHODS or not 0 < self.top_k <= self.latent_dim:
            raise ValueError("Invalid region model configuration")
        self.encoder = nn.Linear(self.input_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, self.input_dim)
        with torch.no_grad():
            nn.init.normal_(self.decoder.weight, std=self.input_dim ** -0.5)
            self.normalize_decoder()
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()
            self.decoder.bias.zero_()

    @torch.no_grad()
    def normalize_decoder(self):
        norms = torch.linalg.vector_norm(self.decoder.weight, dim=0, keepdim=True)
        if torch.any(norms <= 0) or not torch.isfinite(norms).all():
            raise ValueError("Decoder contains an invalid component")
        self.decoder.weight.div_(norms)

    def encode(self, features):
        activations = F.relu(self.encoder(features))
        if self.method.startswith("dense_"):
            return activations
        values, indices = torch.topk(activations, self.top_k, dim=-1, sorted=False)
        return torch.zeros_like(activations).scatter(-1, indices, values)

    @staticmethod
    def memory(codes):
        return F.normalize(codes, p=2, dim=-1)

    def pair_scores(self, source_codes, query_codes):
        return (self.memory(source_codes) * self.memory(query_codes)).sum(-1)


class RegionIdentityPredictor(AcknowledgementPredictor):
    @torch.no_grad()
    def encode_raw(self, raw, batch_size=512):
        values = self._input(raw)
        output = []
        for start in range(0, len(values), batch_size):
            codes = self.model.encode(torch.from_numpy(values[start:start + batch_size]).to(self.device))
            if not torch.isfinite(codes).all() or torch.any(torch.linalg.vector_norm(codes, dim=-1) <= 0):
                raise ValueError("Invalid region latent representation")
            output.append(codes.cpu().numpy())
        return np.concatenate(output) if output else np.empty((0, self.model.latent_dim), dtype=np.float32)

    @torch.no_grad()
    def decode_raw(self, raw, batch_size=512):
        values = self._input(raw)
        output = []
        for start in range(0, len(values), batch_size):
            codes = self.model.encode(torch.from_numpy(values[start:start + batch_size]).to(self.device))
            decoded = self.model.decoder(codes).cpu().numpy().astype(np.float64)
            output.append(decoded * self.scale + self.mean)
        result = np.concatenate(output) if output else np.empty((0, self.model.input_dim), dtype=np.float64)
        if not np.isfinite(result).all():
            raise ValueError("Invalid decoded raw descriptor")
        return result


def load_predictor(fit_directory, device="cpu"):
    directory = Path(fit_directory)
    model = RegionIdentitySAE(read_json(directory / "model_config.json"))
    with np.load(directory / "model.npz", allow_pickle=False) as archive:
        model.load_state_dict({key: torch.from_numpy(archive[key].copy()) for key in archive.files}, strict=True)
    with np.load(directory / "normalization.npz", allow_pickle=False) as archive:
        mean, scale = archive["mean"].copy(), archive["scale"].copy()
    return RegionIdentityPredictor(model, mean, scale, device)
