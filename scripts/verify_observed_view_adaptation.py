import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_observed_view_adaptation as experiment
from src.checkpoint_io import atomic_write_json, read_json


def verify(run):
    config = read_json(run / "config.json")
    recovery = run / "recovery"
    recovery.mkdir()
    atomic_write_json(recovery / "config.json", dict(config, run_dir=str(recovery)))
    shutil.copy2(run / "protocol.json", recovery / "protocol.json")
    command = [sys.executable, "scripts/evaluate_observed_view_adaptation.py", "--run", str(recovery),
        "--phase", "evaluate", "--seed", str(config["seeds"][0]), "--smoke"]
    with (recovery / "execution.log").open("w", encoding="utf-8") as stream:
        first = subprocess.run(command + ["--stop-after-sources", "1"], stdout=stream, stderr=subprocess.STDOUT)
        if first.returncode != 75:
            raise ValueError("Actual source-boundary interruption failed")
        files = sorted((recovery / "smoke/evaluation").rglob("adaptation.npz"))
        if len(files) != 1:
            raise ValueError("Interruption did not preserve exactly one source")
        signature = experiment.parent.digest(files[0])
        second = subprocess.run(command + ["--resume"], stdout=stream, stderr=subprocess.STDOUT)
        if second.returncode != 0 or experiment.parent.digest(files[0]) != signature:
            raise ValueError("Resume changed the completed source or failed")
    count = 0
    for path in (run / "smoke/evaluation").rglob("*.npz"):
        restored = recovery / "smoke/evaluation" / path.relative_to(run / "smoke/evaluation")
        with np.load(path, allow_pickle=False) as a, np.load(restored, allow_pickle=False) as b:
            if a.files != b.files:
                raise ValueError("Resumed keys differ")
            for name in a.files:
                if not np.array_equal(a[name], b[name], equal_nan=a[name].dtype.kind in "fc"):
                    raise ValueError(f"Resumed array differs: {path}/{name}")
                count += 1
    records = read_json(Path(config["parent_dictionary_run"]) / "descriptor_records.json")
    checks = []
    for method in config["methods"]:
        model = experiment.previous.bundle(config, records, method, config["seeds"][0], "full_training")
        predictor = model["predictor"]
        for path in (run / "smoke/evaluation").rglob("adaptation.npz"):
            with np.load(path, allow_pickle=False) as saved:
                raw, codes = saved["original_raw"], saved[method + "__pooled_codes"]
                gains = saved[method + "__gains"]
                coefficients = (gains * codes).astype(float)
                decoder = predictor.dictionary.decoder.weight.detach().numpy().T.astype(float)
                direct = raw[None] + (coefficients @ decoder) * predictor.token_scale
                terms = np.count_nonzero(coefficients, axis=1)[:, None]
                gamma = terms * np.finfo(np.float32).eps / (1 - terms * np.finfo(np.float32).eps)
                bound = gamma * (np.abs(coefficients) @ np.abs(decoder)) * predictor.token_scale + 1e-12 * np.maximum(1., np.abs(raw))
                if np.any(np.abs(direct - saved[method + "__candidates_raw"]) > bound):
                    raise ValueError("Residual edit exceeds the float32 summation bound")
                memories = predictor.encode(direct)
                error = float(np.max(np.abs(memories - saved[method + "__candidates_memory"])))
                positive = predictor.encode(saved["observed_raw"])
                low = np.quantile(np.clip(memories.astype(float) @ positive.astype(float).T, -1, 1), .1, axis=1)
                tail = np.array([np.quantile(np.clip(model["bank"].astype(float) @ m.astype(float), -1, 1), .99,
                    weights=model["weights"], method="inverted_cdf") for m in memories])
                expected = low - tail - (low[0] - tail[0])
                margin_error = float(np.max(np.abs(expected - saved[method + "__candidate_objective"])))
                if error > 1e-6 or margin_error > 1e-6 or not np.array_equal(raw, saved[method + "__candidates_raw"][0]):
                    raise ValueError("Direct observed-margin verification failed")
                source = experiment.previous.adaptation.raw_key(raw)
                wrapper = experiment.previous.adaptation.TemplatePredictor(
                    experiment.parent.reference_api.Representations(Path(config["reference_fit"])),
                    {source: (raw, saved["memory__view_svm"])}, True)
                queries = model["assets"]["raw"][:32, 0]
                vector = saved["memory__view_svm"]
                from scipy.special import expit
                expected_svm = expit(predictor.encode(queries).astype(float) @ vector[:128] + vector[128])
                svm_error = float(np.max(np.abs(expected_svm - wrapper.score(raw, queries))))
                if svm_error > 1e-6:
                    raise ValueError("SVM query scores differ")
                checks.append(dict(method=method, source=str(path), memory_error=error, margin_error=margin_error,
                    svm_error=svm_error, zero_exact=True, fp32_bound=True, observed_frames=saved["observed_frames"].tolist()))
    atomic_write_json(recovery / "verification.json", dict(status="PASS", interrupted_exit_code=75,
        first_source_preserved=True, arrays_exact=count, checks=checks, source_sha256=experiment.parent.digest(__file__)))
    print("VERIFIED", count, checks, flush=True)


def verify_multiframe(run):
    config = read_json(run / "config.json")
    settings = read_json(config["evaluation_settings_file"])
    records = experiment.parent.source_records(config, settings, False)
    inventory = read_json(run / "source_support_inventory.json")["sources"]
    selected = []
    for population in ("development", "extension"):
        episode = next(r["episode"] for r in inventory if r["population"] == population
            and np.count_nonzero(r["frame_counts"]) > 1)
        selected.append(next(r for r in records if r["episode"]["episode_id"] == episode))
    training = read_json(Path(config["parent_dictionary_run"]) / "descriptor_records.json")
    seed = config["seeds"][0]
    models = {m: experiment.previous.bundle(config, training, m, seed, "full_training") for m in config["methods"]}
    output = run / "multiframe_verification"
    output.mkdir()
    reference = experiment.parent.reference_api.Representations(Path(config["reference_fit"]))
    experiment.source_memories(run, config, selected, models, reference, output, seed, False, None)
    checks = []
    for path in output.rglob("adaptation.npz"):
        with np.load(path, allow_pickle=False) as saved:
            if len(saved["observed_frames"]) <= 1:
                raise ValueError("Multiframe verification lacks observed variation")
            for method, model in models.items():
                positive = model["predictor"].encode(saved["observed_raw"])
                memories = saved[method + "__candidates_memory"].astype(float)
                low = np.quantile(np.clip(memories @ positive.astype(float).T, -1, 1), .1, axis=1)
                tail = np.array([np.quantile(np.clip(model["bank"].astype(float) @ m, -1, 1), .99,
                    weights=model["weights"], method="inverted_cdf") for m in memories])
                objective = low - tail - (low[0] - tail[0])
                error = float(np.max(np.abs(objective - saved[method + "__candidate_objective"])))
                prefix = "sae" if method == "sparse_edit" else "dense"
                if error > 1e-6 or not np.array_equal(saved["memory__" + prefix + "_views"], memories[np.argmax(objective)]):
                    raise ValueError("Multiframe objective or selected memory differs")
                checks.append(dict(method=method, source=str(path), frames=len(positive), objective_error=error))
    atomic_write_json(output / "verification.json", dict(status="PASS", checks=checks,
        source_hashes=experiment.hashes(), verifier_sha256=experiment.parent.digest(__file__)))
    print("MULTIFRAME_VERIFIED", checks, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--multiframe", action="store_true")
    torch.set_num_threads(1)
    args = parser.parse_args()
    (verify_multiframe if args.multiframe else verify)(args.run)
