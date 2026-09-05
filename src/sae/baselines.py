"""Auditable SAE baseline primitives.

BatchTopK semantics are adapted from saprmarks/dictionary_learning commit
60ec6bf5264944d64a4ca271f45a29ebfb9d4946 (MIT), specifically
dictionary_learning/trainers/batch_top_k.py, SHA-256
8b251f5db27239b82e49d11436ce4463e36606a33188c7d06a65657555835853.

This module intentionally contains no training loop, data loader, EndoFM code,
or optional language-model dependencies. Experiment runners own optimization,
provenance, progress reporting, and checkpoint policy.
"""

from __future__ import annotations

import torch
import torch.autograd as autograd
from torch import nn


def _sum_to_shape(value: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Reduce broadcast dimensions so custom gradients match a parameter shape."""
    while value.ndim > len(shape):
        value = value.sum(dim=0)
    for axis, size in enumerate(shape):
        if size == 1 and value.shape[axis] != 1:
            value = value.sum(dim=axis, keepdim=True)
    return value


class _RectangleSTE(autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        (x,) = ctx.saved_tensors
        return (grad_output * ((x > -0.5) & (x < 0.5)).to(grad_output.dtype),)


class _JumpReLUSTE(autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float
    ) -> torch.Tensor:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = float(bandwidth)
        ctx.threshold_shape = threshold.shape
        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        x_grad = (x > threshold).to(grad_output.dtype) * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * _RectangleSTE.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, _sum_to_shape(threshold_grad, ctx.threshold_shape), None


class _StepSTE(autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float
    ) -> torch.Tensor:
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = float(bandwidth)
        ctx.threshold_shape = threshold.shape
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth
        threshold_grad = (
            -(1.0 / bandwidth)
            * _RectangleSTE.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return torch.zeros_like(x), _sum_to_shape(threshold_grad, ctx.threshold_shape), None


def differentiable_l0(
    preactivations: torch.Tensor, threshold: torch.Tensor, bandwidth: float
) -> torch.Tensor:
    """Batch-mean feature count with the audited Heaviside threshold STE."""
    return differentiable_l0_per_example(preactivations, threshold, bandwidth).mean()


def differentiable_l0_per_example(
    preactivations: torch.Tensor, threshold: torch.Tensor, bandwidth: float
) -> torch.Tensor:
    """Per-example feature counts required by JumpReLU Appendix-F Eq. 40."""
    if preactivations.ndim != 2 or threshold.shape != (preactivations.shape[1],):
        raise ValueError("preactivations/threshold shape mismatch")
    return _StepSTE.apply(preactivations, threshold, bandwidth).sum(dim=-1)


class JumpReLUAutoencoder(nn.Module):
    """JumpReLU SAE with positive per-feature thresholds parameterized in log space.

    The activation and Heaviside backward estimators follow Appendix J of
    Rajamanoharan et al. (2024). This class exposes primitives only; runners own
    normalization, schedules, optimization, rate selection, and provenance.
    """

    def __init__(
        self,
        activation_dim: int,
        dictionary_size: int,
        *,
        initial_threshold: float = 0.1,
        bandwidth: float = 0.01,
    ) -> None:
        super().__init__()
        if activation_dim <= 0 or dictionary_size <= 0:
            raise ValueError("activation_dim and dictionary_size must be positive")
        if initial_threshold <= 0 or bandwidth <= 0:
            raise ValueError("initial_threshold and bandwidth must be positive")
        self.activation_dim = activation_dim
        self.dictionary_size = dictionary_size
        self.bandwidth = float(bandwidth)
        self.decoder = nn.Linear(dictionary_size, activation_dim, bias=False)
        self.encoder = nn.Linear(activation_dim, dictionary_size)
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))
        self.log_threshold = nn.Parameter(
            torch.full((dictionary_size,), float(torch.tensor(initial_threshold).log()))
        )
        self.normalize_decoder_()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()

    @property
    def threshold(self) -> torch.Tensor:
        return self.log_threshold.exp()

    def preactivations(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x - self.b_dec))

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = self.preactivations(x)
        return _JumpReLUSTE.apply(pre, self.threshold, self.bandwidth), pre

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return self.decoder(code) + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code, pre = self.encode(x)
        return self.decode(code), code, pre

    def differentiable_l0(self, preactivations: torch.Tensor) -> torch.Tensor:
        return differentiable_l0(preactivations, self.threshold, self.bandwidth)

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12)
        self.decoder.weight.div_(norms)

    @torch.no_grad()
    def project_decoder_gradient_(self) -> None:
        if self.decoder.weight.grad is None:
            raise RuntimeError("decoder gradient is unavailable")
        directions = self.decoder.weight / self.decoder.weight.norm(
            dim=0, keepdim=True
        ).clamp_min(1e-12)
        parallel = (
            self.decoder.weight.grad * directions
        ).sum(dim=0, keepdim=True) * directions
        self.decoder.weight.grad.sub_(parallel)


def jumprelu_target_l0_objective(
    x: torch.Tensor,
    reconstruction: torch.Tensor,
    differentiable_l0: torch.Tensor,
    *,
    target_l0: float,
    sparsity_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Appendix-F target-L0 loss with per-example summed reconstruction error."""
    if target_l0 <= 0 or sparsity_coefficient < 0:
        raise ValueError("target_l0 must be positive and sparsity_coefficient nonnegative")
    reconstruction_loss = (x - reconstruction).square().sum(dim=-1).mean()
    rate_loss = sparsity_coefficient * (
        differentiable_l0 / target_l0 - 1.0
    ).square().mean()
    return reconstruction_loss + rate_loss, reconstruction_loss, rate_loss


class BatchTopKAutoencoder(nn.Module):
    """BatchTopK SAE with training-time batch competition and threshold inference."""

    def __init__(self, activation_dim: int, dictionary_size: int, k: int) -> None:
        super().__init__()
        if activation_dim <= 0 or dictionary_size <= 0 or k <= 0:
            raise ValueError("activation_dim, dictionary_size, and k must be positive")
        if k > dictionary_size:
            raise ValueError("k cannot exceed dictionary_size")
        self.activation_dim = activation_dim
        self.dictionary_size = dictionary_size
        self.register_buffer("k", torch.tensor(k, dtype=torch.int64))
        self.register_buffer("threshold", torch.tensor(-1.0, dtype=torch.float32))
        self.decoder = nn.Linear(dictionary_size, activation_dim, bias=False)
        self.encoder = nn.Linear(activation_dim, dictionary_size)
        self.b_dec = nn.Parameter(torch.zeros(activation_dim))
        self.normalize_decoder_()
        with torch.no_grad():
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()

    def preactivations(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x - self.b_dec))

    def encode_training(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = self.preactivations(x)
        flat = pre.flatten()
        count = int(self.k.item()) * x.shape[0]
        selected = flat.topk(count, sorted=False)
        code = torch.zeros_like(flat).scatter(0, selected.indices, selected.values).reshape_as(pre)
        return code, pre

    def encode_inference(self, x: torch.Tensor) -> torch.Tensor:
        if self.threshold < 0:
            raise RuntimeError("inference threshold has not been estimated from training data")
        pre = self.preactivations(x)
        return pre * (pre > self.threshold)

    def decode(self, code: torch.Tensor) -> torch.Tensor:
        return self.decoder(code) + self.b_dec

    def forward(self, x: torch.Tensor, *, inference: bool) -> tuple[torch.Tensor, torch.Tensor]:
        code = self.encode_inference(x) if inference else self.encode_training(x)[0]
        return self.decode(code), code

    @torch.no_grad()
    def initialize_decoder_bias_geometric_median_(
        self, points: torch.Tensor, max_iter: int = 100, tolerance: float = 1e-5
    ) -> None:
        guess = points.mean(dim=0)
        for _ in range(max_iter):
            distances = torch.linalg.vector_norm(points - guess, dim=1).clamp_min(1e-12)
            weights = distances.reciprocal()
            weights = weights / weights.sum()
            updated = (weights[:, None] * points).sum(dim=0)
            if torch.linalg.vector_norm(updated - guess) < tolerance:
                guess = updated
                break
            guess = updated
        self.b_dec.copy_(guess.to(self.b_dec.dtype))

    @torch.no_grad()
    def update_threshold_(self, training_code: torch.Tensor, beta: float) -> None:
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must lie in [0, 1)")
        active = training_code[training_code > 0]
        minimum = active.min().float() if active.numel() else training_code.new_tensor(0.0).float()
        if self.threshold < 0:
            self.threshold.copy_(minimum)
        else:
            self.threshold.mul_(beta).add_(minimum * (1.0 - beta))

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12)
        self.decoder.weight.div_(norms)

    @torch.no_grad()
    def project_decoder_gradient_(self) -> None:
        if self.decoder.weight.grad is None:
            raise RuntimeError("decoder gradient is unavailable")
        directions = self.decoder.weight / self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12)
        parallel = (self.decoder.weight.grad * directions).sum(dim=0, keepdim=True) * directions
        self.decoder.weight.grad.sub_(parallel)


class TopKAutoencoder(BatchTopKAutoencoder):
    """Per-sample TopK-slot control sharing the audited decoder and AuxK interface.

    ReLU is applied before index selection. The operator therefore always selects
    ``k`` slots, but mathematical nonzero L0 can be below ``k`` when fewer than
    ``k`` preactivations are strictly positive. Callers must measure both notions
    rather than describe this implementation as exact-nonzero-L0.
    """

    def encode_training(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = self.preactivations(x)
        selected = pre.topk(int(self.k.item()), dim=-1, sorted=False)
        code = torch.zeros_like(pre).scatter(1, selected.indices, selected.values)
        return code, pre

    def encode_inference(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode_training(x)[0]


def normalized_auxk_loss(
    residual: torch.Tensor,
    preactivations: torch.Tensor,
    decoder_weight: torch.Tensor,
    dead_features: torch.Tensor,
    k_aux: int,
) -> torch.Tensor:
    """Reconstruct detached residual with the strongest currently dead features."""
    if dead_features.dtype is not torch.bool or dead_features.ndim != 1:
        raise ValueError("dead_features must be a one-dimensional boolean tensor")
    dead_count = int(dead_features.sum().item())
    if dead_count == 0 or k_aux <= 0:
        return residual.new_tensor(0.0)
    chosen = min(k_aux, dead_count)
    candidates = torch.where(dead_features[None], preactivations, -torch.inf)
    values, indices = candidates.topk(chosen, dim=-1, sorted=False)
    code = torch.zeros_like(preactivations).scatter(1, indices, values)
    reconstruction = torch.nn.functional.linear(code, decoder_weight)
    numerator = (residual.float() - reconstruction.float()).square().sum(dim=-1).mean()
    centered = residual.float() - residual.float().mean(dim=0, keepdim=True)
    denominator = centered.square().sum(dim=-1).mean()
    return (numerator / denominator).nan_to_num(0.0)
