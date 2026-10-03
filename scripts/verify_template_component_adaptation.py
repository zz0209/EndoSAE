import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_template_component_adaptation as experiment
from src.checkpoint_io import atomic_write_json, read_json


def verify(run):
    torch.set_num_threads(1)
    config = read_json(run / "config.json")
    recovery = run / "recovery_v2"
    recovery.mkdir(exist_ok=False)
    atomic_write_json(recovery / "config.json", config)
    atomic_write_json(recovery / "protocol.json", read_json(run / "protocol.json"))
    command = [sys.executable, str(ROOT / "scripts/evaluate_template_component_adaptation.py"), "--run", str(recovery),
               "--phase", "evaluate", "--seed", str(config["seeds"][0]), "--smoke"]
    with (recovery / "interruption.log").open("w") as log:
        interrupted = subprocess.run(command + ["--stop-after-sources", "1"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if interrupted.returncode != 75:
        raise ValueError(f"Expected actual source-boundary exit75, got {interrupted.returncode}")
    first = next((recovery / "smoke/evaluation").rglob("adaptation.npz"))
    first_hash = experiment.parent.digest(first)
    with (recovery / "resume.log").open("w") as log:
        subprocess.run(command + ["--resume"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    if experiment.parent.digest(first) != first_hash:
        raise ValueError("Completed source checkpoint changed during resume")
    atomic_write_json(recovery / "execution_check.json", dict(actual_interruption_returncode=75,
        first_checkpoint_preserved=True, first_checkpoint_sha256=first_hash))
    arrays = 0
    for path in (run / "smoke/evaluation").rglob("*.npz"):
        other = recovery / "smoke/evaluation" / path.relative_to(run / "smoke/evaluation")
        with np.load(path, allow_pickle=False) as expected, np.load(other, allow_pickle=False) as actual:
            if expected.files != actual.files:
                raise ValueError("Resumed array keys differ")
            for name in expected.files:
                if not np.array_equal(expected[name], actual[name], equal_nan=expected[name].dtype.kind in "fc"):
                    raise ValueError(f"Resumed values differ at {path}/{name}")
                arrays += 1
    records = read_json(Path(config["parent_dictionary_run"]) / "descriptor_records.json")
    checks = []
    for method in config["methods"]:
        model = experiment.bundle(config, records, method, config["seeds"][0], "full_training")
        predictor = model["predictor"]
        for path in (run / "smoke/evaluation").rglob("adaptation.npz"):
            with np.load(path, allow_pickle=False) as saved:
                raw, codes = saved["original_raw"], saved[method + "__pooled_codes"]
                active = saved[method + "__active_components"]
                gains = saved[method + "__gains"]
                memories = saved[method + "__candidates_memory"]
                coefficients = (gains * codes).astype(np.float64)
                decoder = predictor.dictionary.decoder.weight.detach().numpy().T.astype(np.float64)
                direct = raw[None] + (coefficients @ decoder) * predictor.token_scale
                raw_error = float(np.max(np.abs(direct - saved[method + "__candidates_raw"])))
                terms = np.count_nonzero(coefficients, axis=1)[:, None]
                gamma = terms * np.finfo(np.float32).eps / (1 - terms * np.finfo(np.float32).eps)
                bound = gamma * (np.abs(coefficients) @ np.abs(decoder)) * predictor.token_scale + 1e-12 * np.maximum(1., np.abs(raw))
                if np.any(np.abs(direct - saved[method + "__candidates_raw"]) > bound):
                    raise ValueError("Grouped edit exceeds the float32 summation error bound")
                memory_error = float(np.max(np.abs(predictor.encode(direct) - memories)))
                if memory_error > 1e-6:
                    raise ValueError("Independent float64 edit changes identity memory")
                component = int(active[0])
                single = raw - (codes[component] * predictor.dictionary.decoder.weight.detach().numpy()[:, component]).astype(float) * predictor.token_scale
                base, edited = predictor.encode(np.vstack([raw, single]))
                before = np.quantile(np.clip(model["bank"].astype(float) @ base.astype(float), -1, 1), .99,
                    method="inverted_cdf", weights=model["weights"])
                after = np.quantile(np.clip(model["bank"].astype(float) @ edited.astype(float), -1, 1), .99,
                    method="inverted_cdf", weights=model["weights"])
                benefit_error = float(abs(before - after - saved[method + "__component_benefit"][component]))
                if benefit_error > 1e-6:
                    raise ValueError("Single-component effect differs from direct re-encoding")
                if not np.array_equal(saved[method + "__candidates_raw"][0], raw):
                    raise ValueError("Zero action altered the source")
                query_raw = model["assets"]["raw"][:32, 0]
                wrapper_reference = experiment.parent.reference_api.Representations(Path(config["reference_fit"]))
                svm = saved["memory__template_svm"]
                wrapped = experiment.adaptation.TemplatePredictor(wrapper_reference,
                    {experiment.adaptation.raw_key(raw): (raw, svm)}, True)
                direct_score = expit(predictor.encode(query_raw).astype(float) @ svm[:128] + svm[128])
                score_error = float(np.max(np.abs(direct_score - wrapped.score(raw, query_raw))))
                if score_error > 1e-6:
                    raise ValueError("Template SVM query score differs between encoders")
                checks.append(dict(method=method, source=str(path), group_raw_max_error=raw_error,
                    fp32_error_bound_pass=True, group_memory_max_error=memory_error,
                    component=component, component_benefit_error=benefit_error, svm_score_error=score_error))
    atomic_write_json(recovery / "verification.json", dict(status="PASS", resumed_arrays=arrays,
        first_checkpoint_preserved=True, actual_interruption_returncode=75, direct_checks=checks,
        source_sha256=experiment.parent.digest(__file__)))
    print("VERIFIED", arrays, checks, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    verify(parser.parse_args().run)
