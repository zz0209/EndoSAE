import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_io import atomic_write_json, read_json
from src.shared_component_editor import correction_loss, edit_scores


def numpy_scores(first, second, a, b, directions, gains, budget, mode):
    common = np.where(a * b > 0, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0)
    coefficients = common if mode == "bilateral" else a
    delta = (coefficients * gains) @ directions.T
    lengths = np.linalg.norm(delta, axis=1)
    scale = np.ones(len(lengths))
    outside = lengths > budget
    scale[outside] = budget / lengths[outside]
    delta *= scale[:, None]
    u, v = first - delta, second - delta if mode == "bilateral" else second
    scores = np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    if mode == "bilateral":
        dot = np.sum(first * second, axis=1)
        left, right = np.sum(first * delta, axis=1), np.sum(second * delta, axis=1)
        squared = np.sum(delta * delta, axis=1)
        formula = (dot - left - right + squared) / np.sqrt(
            (np.sum(first * first, axis=1) - 2 * left + squared) *
            (np.sum(second * second, axis=1) - 2 * right + squared))
        np.testing.assert_allclose(scores, formula, atol=1e-12, rtol=1e-12)
    return scores, delta


def verify(run, directory, recovery):
    torch.set_num_threads(1)
    config = read_json(run / "config.json")
    summary = read_json(directory / "training_summary.json")
    checks = []
    for row in summary["outputs"]:
        folder = Path(row["directory"])
        with np.load(Path(row.get("basis_directory", folder.parent)) / "basis.npz", allow_pickle=False) as basis, np.load(folder / "gains.npz", allow_pickle=False) as weights:
            embedding, code, directions = (basis[key].astype(np.float64) for key in ["embedding", "code", "directions"])
            gains = weights["gains"].astype(np.float64)
        for partition in ["fit", "held"]:
            with np.load(folder / f"{partition}.npz", allow_pickle=False) as saved:
                first, second = embedding[saved["source"], saved["view"]], embedding[saved["query"], 0]
                a, b = code[saved["source"], saved["view"]], code[saved["query"], 0]
                actual, delta = numpy_scores(first, second, a, b, directions, gains, config["edit_budget"], row["mode"])
                error = float(np.max(np.abs(actual - saved["after"])))
                if error > 2e-6:
                    raise ValueError("Independent shared-component forward mismatch")
                zero, _ = numpy_scores(first, second, a, b, directions, np.zeros_like(gains), config["edit_budget"], row["mode"])
                np.testing.assert_allclose(zero, saved["before"], atol=2e-6, rtol=0)
                np.testing.assert_allclose(np.linalg.norm(delta, axis=1), saved["edit_norm"], atol=2e-6, rtol=0)
                if "parent_policy" in row:
                    with np.load(Path(row["parent_policy"]) / "gains.npz", allow_pickle=False) as original:
                        reference, _ = numpy_scores(first, second, a, b, directions,
                            original["gains"].astype(np.float64), config["edit_budget"], row["mode"])
                    np.testing.assert_allclose(reference, saved["in_sample"], atol=2e-6, rtol=0)
                    if partition == "held":
                        with np.load(Path(row["parent_policy"]) / "held.npz", allow_pickle=False) as original:
                            positions = {(int(s), int(q), int(v)): i for i, (s, q, v) in enumerate(
                                zip(original["source"], original["query"], original["view"]))}
                            take = [positions[(int(s), int(q), int(v))] for s, q, v in
                                zip(saved["source"], saved["query"], saved["view"])]
                            np.testing.assert_allclose(saved["in_sample"], original["after"][take], atol=2e-6, rtol=0)
                            np.testing.assert_array_equal(saved["labels"], original["labels"][take])
                if row["mode"] == "bilateral":
                    swapped, _ = numpy_scores(second, first, b, a, directions, gains, config["edit_budget"], row["mode"])
                    np.testing.assert_allclose(actual, swapped, atol=1e-12, rtol=0)
                gradient_error = None
                if partition == "fit":
                    threshold = (row["original_threshold"] if "original_threshold" in row else
                        row["receipt"]["reports"]["fit"]["before_threshold"])
                    labels, fitting_weights = saved["labels"].copy(), saved["weights"].astype(np.float64)
                    interior = .1 + .8 * gains
                    g = torch.tensor(interior, requires_grad=True, dtype=torch.float64)
                    values = [torch.from_numpy(x) for x in [first, second, a, b, directions]]
                    scores, _, _ = edit_scores(*values, g, config["edit_budget"], row["mode"])
                    objective = correction_loss(scores, torch.from_numpy(labels), torch.from_numpy(fitting_weights),
                        threshold, config["temperature"], g, config["gain_penalty"])
                    objective.backward()
                    rng = np.random.default_rng(19337)
                    direction = rng.normal(size=len(gains))
                    direction /= np.linalg.norm(direction)
                    derivatives = []
                    for step in [-1e-6, 1e-6]:
                        candidate = interior + step * direction
                        prediction, _ = numpy_scores(first, second, a, b, directions, candidate, config["edit_budget"], row["mode"])
                        signed = np.where(labels, threshold - prediction, prediction - threshold) / config["temperature"]
                        derivatives.append(np.sum(fitting_weights * np.logaddexp(0, signed)) + config["gain_penalty"] * candidate.mean())
                    numerical = (derivatives[1] - derivatives[0]) / 2e-6
                    analytic = float(g.grad.numpy() @ direction)
                    gradient_error = abs(numerical - analytic)
                    np.testing.assert_allclose(numerical, analytic, atol=2e-7, rtol=2e-4)
                checks.append(dict(basis=row["basis"], mode=row["mode"], fold=row["fold"], partition=partition,
                    observations=len(actual), forward_max_error=error, directional_gradient_error=gradient_error))
    recovery_arrays = 0
    if recovery:
        for source in sorted(directory.glob("models/**/*.npz")):
            counterpart = recovery / source.relative_to(directory)
            with np.load(source, allow_pickle=False) as a, np.load(counterpart, allow_pickle=False) as b:
                if a.files != b.files:
                    raise ValueError("Recovery keys differ")
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
                    recovery_arrays += 1
    atomic_write_json(run / "numerical_verification.json", dict(status="PASS", checks=checks,
        recovery_arrays_exact=recovery_arrays, actual_data=True))
    print("VERIFIED", len(checks), "forward/formula checks; recovery arrays", recovery_arrays, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--recovery", type=Path)
    args = parser.parse_args()
    verify(args.run, args.directory, args.recovery)
