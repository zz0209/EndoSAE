from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.checkpoint_io import read_json
from src.region_identity_sae import RegionIdentitySAE


METHODS = ("sparse_edit", "dense_edit")


class TokenDictionary(RegionIdentitySAE):
    def __init__(self, config):
        if config["method"] not in METHODS:
            raise ValueError("Unknown token dictionary method")
        super().__init__(dict(config, method="sparse_self" if config["method"] == "sparse_edit" else "dense_self"))
        self.edit_method = config["method"]


class BoundedGains(nn.Module):
    def __init__(self, latent_dim, bound):
        super().__init__()
        self.bound = float(bound)
        if not 0 < self.bound <= 1:
            raise ValueError("Gain bound must be in (0,1]")
        self.theta = nn.Parameter(torch.zeros(int(latent_dim)))

    def forward(self):
        return self.bound * torch.tanh(self.theta)


class FrozenSupCon(nn.Module):
    def __init__(self, directory):
        super().__init__()
        directory = Path(directory)
        with np.load(directory / "normalization.npz", allow_pickle=False) as data:
            self.register_buffer("mean", torch.from_numpy(data["mean"].copy()).double())
            self.register_buffer("scale", torch.from_numpy(data["scale"].copy()).double())
        self.network = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Linear(256, 128))
        with np.load(directory / "model.npz", allow_pickle=False) as data:
            self.network.load_state_dict({key: torch.from_numpy(data[key].copy()) for key in data.files}, strict=True)
        self.eval().requires_grad_(False)

    def forward(self, raw):
        standardized = ((raw.double() - self.mean) / self.scale).float()
        return F.normalize(self.network(standardized), dim=-1)


def residual_edit(raw, mean_codes, gains, dictionary, token_scale):
    correction = ((mean_codes.float() * gains) @ dictionary.decoder.weight.T).double()
    return raw.double() + correction * token_scale.double()


class TokenMemoryPredictor:
    def __init__(self, dictionary, token_mean, token_scale, gains, reference, device="cpu"):
        self.device = torch.device(device)
        self.dictionary = dictionary.to(self.device).eval().requires_grad_(False)
        self.token_mean = np.asarray(token_mean, dtype=np.float64)
        self.token_scale = np.asarray(token_scale, dtype=np.float64)
        self.source_gains = np.asarray(gains, dtype=np.float32)
        self.reference = reference.to(self.device).eval().requires_grad_(False)
        if (self.token_mean.shape != (dictionary.input_dim,) or self.token_scale.shape != self.token_mean.shape
                or self.source_gains.shape != (dictionary.latent_dim,)):
            raise ValueError("Token edit parameter dimensions disagree")
        if (not np.isfinite(self.token_mean).all() or not np.isfinite(self.token_scale).all()
                or np.any(self.token_scale <= 0) or not np.isfinite(self.source_gains).all()):
            raise ValueError("Invalid token edit parameters")
        self.decoder_raw_directions = dictionary.decoder.weight.detach().cpu().numpy().T.astype(np.float64) * self.token_scale

    @torch.no_grad()
    def encode_tokens(self, tokens, batch_size=2048):
        tokens = np.asarray(tokens)
        if tokens.shape[-1] != self.dictionary.input_dim or not np.isfinite(tokens).all():
            raise ValueError("Invalid raw token array")
        shape = tokens.shape[:-1]
        flat = tokens.reshape(-1, self.dictionary.input_dim)
        outputs = []
        for start in range(0, len(flat), batch_size):
            standardized = ((flat[start:start + batch_size].astype(np.float64) - self.token_mean) / self.token_scale).astype(np.float32)
            codes = self.dictionary.encode(torch.from_numpy(standardized).to(self.device))
            outputs.append(codes.cpu().numpy())
        result = np.concatenate(outputs) if outputs else np.empty((0, self.dictionary.latent_dim), dtype=np.float32)
        return result.reshape(*shape, self.dictionary.latent_dim)

    def pooled_codes(self, tokens, mask):
        tokens, mask = np.asarray(tokens), np.asarray(mask)
        if mask.dtype != np.bool_ or mask.shape != tokens.shape[:-1] or not mask.any():
            raise ValueError("Token source mask must contain observed support")
        return self.encode_tokens(tokens[mask]).mean(axis=0, dtype=np.float64).astype(np.float32)

    @torch.no_grad()
    def edit_raw(self, raw_mean, mean_codes, gain_override=None):
        raw, codes = np.asarray(raw_mean), np.asarray(mean_codes)
        gains = self.source_gains if gain_override is None else np.asarray(gain_override, dtype=np.float32)
        if raw.shape[-1] != self.dictionary.input_dim or codes.shape != (*raw.shape[:-1], self.dictionary.latent_dim):
            raise ValueError("Raw source and pooled token code shapes disagree")
        if gains.shape != self.source_gains.shape or not all(np.isfinite(x).all() for x in (raw, codes, gains)):
            raise ValueError("Invalid residual edit input")
        if not np.any(gains):
            return raw.astype(np.float64, copy=True)
        edited = residual_edit(torch.from_numpy(raw.astype(np.float64)).to(self.device),
            torch.from_numpy(codes.astype(np.float32)).to(self.device), torch.from_numpy(gains).to(self.device),
            self.dictionary, torch.from_numpy(self.token_scale).to(self.device))
        return edited.cpu().numpy()

    def source_delta(self, tokens, mask, gain_override=None):
        codes = self.pooled_codes(tokens, mask)
        return self.edit_raw(np.zeros(self.dictionary.input_dim, dtype=np.float64), codes, gain_override)

    def edit_delta(self, tokens, mask, gain_override=None):
        return self.source_delta(tokens, mask, gain_override)

    @torch.no_grad()
    def encode(self, raw, batch_size=512):
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim == 1:
            raw = raw[None]
        if raw.ndim != 2 or raw.shape[1] != 768 or not np.isfinite(raw).all():
            raise ValueError("Invalid frozen-reference query descriptors")
        result = [self.reference(torch.from_numpy(raw[start:start + batch_size]).to(self.device)).cpu().numpy()
                  for start in range(0, len(raw), batch_size)]
        return np.concatenate(result) if result else np.empty((0, 128), dtype=np.float32)

    def memory_from_pooled(self, original_raw, mean_codes, gain_override=None):
        edited = self.edit_raw(original_raw, mean_codes, gain_override)
        result = self.encode(edited)
        if result.shape != (1, 128):
            raise ValueError("A source memory requires exactly one descriptor")
        return result[0]

    def memory(self, original_raw, tokens, mask, gain_override=None):
        return self.memory_from_pooled(original_raw, self.pooled_codes(tokens, mask), gain_override)

    @staticmethod
    def score_encoded(memory, queries):
        memory, queries = np.asarray(memory, dtype=np.float32), np.asarray(queries, dtype=np.float32)
        return queries @ memory


def load_predictor(fit_directory, device="cpu"):
    directory = Path(fit_directory)
    config = read_json(directory / "model_config.json")
    dictionary = TokenDictionary(config)
    with np.load(directory / "model.npz", allow_pickle=False) as data:
        dictionary.load_state_dict({key: torch.from_numpy(data[key].copy()) for key in data.files}, strict=True)
    with np.load(directory / "normalization.npz", allow_pickle=False) as data:
        mean, scale = data["mean"].copy(), data["scale"].copy()
    with np.load(directory / "gains.npz", allow_pickle=False) as data:
        gains = data["gains"].copy()
        expected = float(config["gain_bound"]) * np.tanh(data["theta"])
        if not np.allclose(gains, expected, rtol=1e-6, atol=1e-8):
            raise ValueError("Exported gains disagree with bounded parameters")
    return TokenMemoryPredictor(dictionary, mean, scale, gains, FrozenSupCon(config["reference_fit"]), device)
