import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_token_identity_sae import load_inputs, normalize
from train_frozen_identity_supcon import supcon
from src.checkpoint_io import atomic_write_json, read_json
from src.token_identity_sae import TokenIdentitySAE, local_identity_loss, symmetric_maxsim


def numpy_forward(model, values, interaction="pooled_cosine"):
    weights = {key: value.detach().numpy() for key, value in model.state_dict().items()}
    if model.method == "raw_supcon":
        hidden = np.maximum(values.mean(1) @ weights["projection.0.weight"].T + weights["projection.0.bias"], 0)
        output = hidden @ weights["projection.2.weight"].T + weights["projection.2.bias"]
        return output / np.linalg.norm(output, axis=-1, keepdims=True), None

    def encode(value):
        code = np.maximum(value @ weights["encoder.weight"].T + weights["encoder.bias"], 0)
        if model.method.endswith("sparse"):
            indices = np.argsort(code, axis=-1)[..., -model.top_k:]
            selected = np.take_along_axis(code, indices, axis=-1)
            result = np.zeros_like(code)
            np.put_along_axis(result, indices, selected, axis=-1)
            return result
        return code

    local = encode(values)
    pooled = local.mean(1) if model.method.startswith("token_") else encode(values.mean(1))
    identity_input = local if interaction == "symmetric_maxsim" else pooled
    output = identity_input @ weights["readout.weight"].T if model.identity_space == "projected" else identity_input
    decoded = local @ weights["decoder.weight"].T + weights["decoder.bias"]
    return output / np.linalg.norm(output, axis=-1, keepdims=True), decoded


def verify(run, output, recovery):
    torch.set_num_threads(1)
    config, cohort, raw, offsets, records, _ = load_inputs(run)
    groups = {}
    for index, row in enumerate(records):
        if row["partition"] == "train":
            groups.setdefault((row["video_id"], row["lesion_id"]), []).append(index)
    chosen = [index for group in list(groups.values())[:2] for index in group[:2]]
    if len(chosen) != 4:
        raise ValueError("Actual paired fixture unavailable")
    mean, scale = normalize(raw, offsets, records, cohort["fit_video_ids"]["train"])
    samples = np.stack([raw[np.linspace(offsets[i], offsets[i + 1] - 1, 8, dtype=int)] for i in chosen])
    samples = (samples - mean) / scale
    values = torch.from_numpy(samples)
    labels = torch.tensor([0, 0, 1, 1])
    identity = np.array([0, 0, 1, 1])
    checks = []
    initial_states = []
    interaction = config.get("identity_interaction", "pooled_cosine")
    for method in config["methods"]:
        torch.manual_seed(config["seeds"][0])
        initial = TokenIdentitySAE(config, method)
        if method != "raw_supcon":
            initial_states.append(initial.state_dict())
        folder = run / "smoke" / "fit" / method / f"seed{config['seeds'][0]}"
        model = TokenIdentitySAE(config, method).double()
        with np.load(folder / "model.npz", allow_pickle=False) as archive:
            model.load_state_dict({key: torch.from_numpy(archive[key].copy()) for key in archive.files})
        expected, decoded = numpy_forward(model, samples, interaction)
        actual, reconstruction, code = model(values)
        if interaction == "symmetric_maxsim":
            actual = torch.nn.functional.normalize(model.readout(code), dim=-1)
        np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-11, rtol=1e-10)
        if decoded is not None:
            np.testing.assert_allclose(reconstruction.detach().numpy(), decoded, atol=1e-11, rtol=1e-10)
        if interaction == "symmetric_maxsim":
            scores = np.empty((len(expected), len(expected)))
            for i, first in enumerate(expected):
                for j, second in enumerate(expected):
                    similarities = first @ second.T
                    scores[i, j] = .5 * (similarities.max(axis=1).mean() + similarities.max(axis=0).mean())
            observed = symmetric_maxsim(actual[:, None], actual[None, :]).detach().numpy()
            np.testing.assert_allclose(observed, scores, atol=1e-11, rtol=1e-10)
            logits = scores / config["temperature"]
        else:
            logits = expected @ expected.T / config["temperature"]
        np.fill_diagonal(logits, -np.inf)
        log_prob = logits - logsumexp(logits, axis=1, keepdims=True)
        positive = (identity[:, None] == identity[None, :]) & ~np.eye(len(identity), dtype=bool)
        expected_identity = -float(np.mean(log_prob[positive]))
        expected_reconstruction = float(np.mean((decoded - samples) ** 2)) if decoded is not None else 0.
        expected_loss = expected_identity + config["reconstruction_weight"] * expected_reconstruction

        def objective():
            projected, recreated, local = model(values)
            result = (local_identity_loss(model, local, labels, config["temperature"])
                if interaction == "symmetric_maxsim" else supcon(projected, labels, config["temperature"]))
            return result + config["reconstruction_weight"] * (recreated - values).square().mean() if recreated is not None else result

        loss = objective()
        np.testing.assert_allclose(float(loss.detach()), expected_loss, atol=1e-11, rtol=1e-10)
        loss.backward()
        generator = torch.Generator().manual_seed(1741)
        directions = [torch.randn(p.shape, generator=generator, dtype=p.dtype) for p in model.parameters()]
        length = torch.sqrt(sum(direction.square().sum() for direction in directions))
        directions = [direction / length for direction in directions]
        analytic = float(sum((parameter.grad * direction).sum() for parameter, direction in zip(model.parameters(), directions)).detach())
        original = [p.detach().clone() for p in model.parameters()]
        h = 1e-5
        outcomes = []
        for sign in [-1, 1]:
            with torch.no_grad():
                for parameter, value, direction in zip(model.parameters(), original, directions, strict=True):
                    parameter.copy_(value + sign * h * direction)
                outcomes.append(float(objective()))
        numerical = (outcomes[1] - outcomes[0]) / (2 * h)
        np.testing.assert_allclose(analytic, numerical, atol=2e-7, rtol=2e-4)
        checks.append(dict(method=method, forward_error=float(np.max(np.abs(actual.detach().numpy() - expected))),
            loss_error=abs(float(loss.detach()) - expected_loss), analytic_direction=analytic,
            finite_difference=numerical, local_active_max=int((code > 0).sum(-1).max()) if code is not None else None))
    for state in initial_states[1:]:
        if any(not torch.equal(value, initial_states[0][key]) for key, value in state.items()):
            raise ValueError("Factorial arms have different initialization")
    recovery_checks = 0
    if recovery:
        relative = Path("inner/token_sparse") / f"seed{config['seeds'][0]}" / "fold0"
        for name in ["model.npz", "normalization.npz", "held_0012.npz"]:
            with np.load(run / "smoke" / relative / name, allow_pickle=False) as continuous, np.load(recovery / relative / name, allow_pickle=False) as resumed:
                if continuous.files != resumed.files:
                    raise ValueError("Resumed output keys differ")
                for key in continuous.files:
                    np.testing.assert_array_equal(continuous[key], resumed[key])
                    recovery_checks += 1
        if read_json(run / "smoke" / relative / "sequence.json") != read_json(recovery / relative / "sequence.json"):
            raise ValueError("Resumed sampling differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output, dict(status="PASS", actual_clips=chosen, checks=checks,
        identical_factorial_initialization=True, exact_recovery_arrays=recovery_checks))
    print("TOKEN_VERIFICATION_PASS", len(checks), "methods", recovery_checks, "recovery arrays", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--recovery", type=Path)
    args = parser.parse_args()
    verify(args.run, args.output, args.recovery)
