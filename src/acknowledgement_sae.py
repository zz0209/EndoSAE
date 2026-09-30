import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.checkpoint_io import read_json


METHODS = ("sae_memory", "dense_memory", "sae_identity")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AcknowledgementSAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.method = config["method"]
        if self.method not in METHODS:
            raise ValueError(f"Unknown method: {self.method}")
        self.input_dim = int(config["input_dim"])
        self.latent_dim = int(config["latent_dim"])
        self.top_k = int(config["top_k"])
        self.memory_k = int(config["memory_k"])
        if not 0 < self.memory_k <= self.top_k <= self.latent_dim:
            raise ValueError("Require 0 < memory_k <= top_k <= latent_dim")
        self.encoder = nn.Linear(self.input_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, self.input_dim)
        with torch.no_grad():
            nn.init.normal_(self.decoder.weight, std=self.input_dim ** -0.5)
            self.normalize_decoder()
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()
            self.decoder.bias.zero_()
        self.gate = None
        if self.method != "sae_identity":
            self.gate = nn.Sequential(
                nn.Linear(self.latent_dim, int(config["gate_hidden"])),
                nn.ReLU(),
                nn.Linear(int(config["gate_hidden"]), self.latent_dim),
            )
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)
        temperature = float(config["temperature"])
        self.logit_scale = nn.Parameter(
            torch.tensor(float(np.log(1.0 / temperature))),
            requires_grad=self.gate is not None,
        )
        self.logit_bias = nn.Parameter(torch.tensor(-2.5), requires_grad=self.gate is not None)

    @torch.no_grad()
    def normalize_decoder(self):
        norms = torch.linalg.vector_norm(self.decoder.weight, dim=0, keepdim=True)
        if torch.any(norms <= 0):
            raise ValueError("Decoder contains a zero-norm component")
        self.decoder.weight.div_(norms)

    def encode(self, features):
        activations = F.relu(self.encoder(features))
        if self.method == "dense_memory":
            return activations
        values, indices = torch.topk(activations, self.top_k, dim=-1, sorted=False)
        return torch.zeros_like(activations).scatter(-1, indices, values)

    def memory(self, source_codes):
        if self.gate is None:
            return F.normalize(source_codes, p=2, dim=-1)
        weights = torch.sigmoid(self.gate(F.normalize(source_codes, p=2, dim=-1)))
        weighted = source_codes * weights
        values, indices = torch.topk(weighted, self.memory_k, dim=-1, sorted=False)
        selected = torch.zeros_like(weighted).scatter(-1, indices, values)
        return F.normalize(selected, p=2, dim=-1)

    def pair_scores(self, source_codes, query_codes):
        return (self.memory(source_codes) * F.normalize(query_codes, p=2, dim=-1)).sum(-1)

    def pair_logits(self, source_codes, query_codes):
        scale = self.logit_scale.exp().clamp(max=100.0)
        return self.pair_scores(source_codes, query_codes) * scale + self.logit_bias


def supervised_contrastive_loss(codes, labels, temperature):
    unit = F.normalize(codes, p=2, dim=-1)
    logits = unit @ unit.T / temperature
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=codes.device)
    positive = (labels[:, None] == labels[None, :]) & ~diagonal
    usable = positive.any(dim=1)
    if not usable.any():
        raise ValueError("Identity batch contains no positive pairs")
    log_probability = logits - torch.logsumexp(
        logits.masked_fill(diagonal, -torch.inf), dim=1, keepdim=True
    )
    positive_log_probability = torch.where(positive, log_probability, 0.0).sum(dim=1)
    return -(positive_log_probability[usable] / positive.sum(dim=1)[usable]).mean()


class AcknowledgementPredictor:
    def __init__(self, model, mean, scale, device="cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        if self.mean.shape != (model.input_dim,) or self.scale.shape != self.mean.shape:
            raise ValueError("Normalization dimensions disagree with the model")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or np.any(self.scale <= 0):
            raise ValueError("Invalid normalization statistics")

    def _input(self, raw):
        values = np.asarray(raw)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.model.input_dim:
            raise ValueError(f"Expected raw descriptors [N,{self.model.input_dim}]")
        if not np.isfinite(values).all():
            raise ValueError("Non-finite raw descriptors")
        return ((values.astype(np.float64) - self.mean) / self.scale).astype(np.float32)

    @torch.no_grad()
    def encode(self, raw, batch_size=512):
        values = self._input(raw)
        if len(values) == 0:
            return np.empty((0, self.model.latent_dim), dtype=np.float32)
        result = []
        for start in range(0, len(values), batch_size):
            codes = self.model.encode(torch.from_numpy(values[start:start + batch_size]).to(self.device))
            if not torch.isfinite(codes).all() or torch.any(torch.linalg.vector_norm(codes, dim=-1) <= 0):
                raise ValueError("Invalid or empty latent representation")
            result.append(F.normalize(codes, p=2, dim=-1).cpu().numpy())
        return np.concatenate(result)

    @torch.no_grad()
    def memory(self, source_raw):
        values = self._input(source_raw)
        if len(values) != 1:
            raise ValueError("A fixed memory requires exactly one source descriptor")
        codes = self.model.encode(torch.from_numpy(values).to(self.device))
        memory = self.model.memory(codes)[0].cpu().numpy()
        if not np.isfinite(memory).all() or np.linalg.norm(memory) <= 0:
            raise ValueError("Invalid or empty source memory")
        return memory

    @staticmethod
    def score_encoded(memory, query_codes):
        memory = np.asarray(memory, dtype=np.float32)
        query_codes = np.asarray(query_codes, dtype=np.float32)
        if memory.ndim != 1 or query_codes.ndim != 2 or query_codes.shape[1] != len(memory):
            raise ValueError("Memory and query code dimensions disagree")
        if not np.isfinite(memory).all() or not np.isfinite(query_codes).all():
            raise ValueError("Non-finite memory or query codes")
        return (query_codes @ memory).astype(np.float32)

    def score(self, source_raw, query_raw, batch_size=512):
        return self.score_encoded(self.memory(source_raw), self.encode(query_raw, batch_size))

    def component_contributions(self, source_raw, query_raw, batch_size=512):
        return self.encode(query_raw, batch_size) * self.memory(source_raw)[None, :]


def load_predictor(fit_directory, device="cpu"):
    directory = Path(fit_directory)
    config = read_json(directory / "model_config.json")
    model = AcknowledgementSAE(config)
    with np.load(directory / "model.npz", allow_pickle=False) as archive:
        state = {key: torch.from_numpy(archive[key].copy()) for key in archive.files}
    model.load_state_dict(state, strict=True)
    with np.load(directory / "normalization.npz", allow_pickle=False) as archive:
        mean, scale = archive["mean"].copy(), archive["scale"].copy()
    return AcknowledgementPredictor(model, mean, scale, device)
