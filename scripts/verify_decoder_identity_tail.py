import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_decoder_identity_tail as evaluation
import train_decoder_identity_tail as training
from src.checkpoint_io import atomic_write_json, read_json
from src.token_memory_edit import BoundedGains, residual_edit


def compare_arrays(left, right):
    count = 0
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if a.files != b.files:
            raise ValueError("Resumed keys differ")
        for name in a.files:
            if not np.array_equal(a[name], b[name], equal_nan=a[name].dtype.kind in "fc"):
                raise ValueError(f"Resumed array differs: {left}/{name}")
            count += 1
    return count


def verify(run):
    config = read_json(run / "config.json")
    recovery = run / "recovery"
    recovery.mkdir()
    seed, count, interruptions = config["seeds"][0], 0, []
    for fold in (0, "full_training"):
        for method in config["methods"]:
            for arm in config["arms"]:
                command = dict(config=config, output=recovery / "training", method=method, seed=seed,
                    fold=fold, arm=arm, steps=config["smoke_steps"])
                try:
                    training.fit_job(**command, resume=False, stop_after_step=4)
                except SystemExit as error:
                    if error.code != 75:
                        raise
                else:
                    raise ValueError("Training did not stop at the real checkpoint")
                result = training.fit_job(**command, resume=True, stop_after_step=None)
                folder = Path(result["folder"])
                original = run / "smoke" / folder.relative_to(recovery / "training")
                for name in ("model.npz", "gains.npz", "held.npz"):
                    count += compare_arrays(original / name, folder / name)
                interruptions.append(dict(method=method, fold=fold, arm=arm, interrupted_step=4, resumed_step=config["smoke_steps"]))
    command = [sys.executable, "scripts/evaluate_decoder_identity_tail.py", "--run", str(run),
        "--phase", "evaluate", "--seed", str(seed), "--smoke", "--output", str(recovery / "evaluation")]
    with (recovery / "execution.log").open("w", encoding="utf-8") as stream:
        stopped = subprocess.run(command + ["--stop-after-sources", "1"], stdout=stream, stderr=subprocess.STDOUT)
        if stopped.returncode != 75:
            raise ValueError("Application did not stop at the real source checkpoint")
        paths = list((recovery / "evaluation/sources").rglob("memories.npz"))
        if len(paths) != 1:
            raise ValueError("Unexpected source checkpoint count")
        signature = evaluation.parent.digest(paths[0])
        resumed = subprocess.run(command + ["--resume"], stdout=stream, stderr=subprocess.STDOUT)
        if resumed.returncode != 0 or evaluation.parent.digest(paths[0]) != signature:
            raise ValueError("Source recovery failed or changed completed output")
    original = run / "smoke/evaluation" / f"seed{seed}"
    application_count = sum(compare_arrays(path, recovery / "evaluation" / path.relative_to(original))
                            for path in original.rglob("*.npz"))
    models, _ = evaluation.bundles(config, run / "smoke", seed)
    direct_checks = []
    for path in original.rglob("memories.npz"):
        with np.load(path, allow_pickle=False) as saved:
            raw = saved["original_raw"]
            for label, predictor in models.items():
                codes = saved[label + "__codes"]
                coefficients = (codes * predictor.source_gains).astype(float)
                decoder = predictor.dictionary.decoder.weight.detach().numpy().T.astype(float)
                direct = raw + (coefficients @ decoder) * predictor.token_scale
                terms = np.count_nonzero(coefficients)
                gamma = terms * np.finfo(np.float32).eps / (1 - terms * np.finfo(np.float32).eps)
                bound = gamma * (np.abs(coefficients) @ np.abs(decoder)) * predictor.token_scale + 1e-12 * np.maximum(1, np.abs(raw))
                if np.any(np.abs(direct - saved[label + "__edited_raw"]) > bound):
                    raise ValueError("Decoder edit exceeds its FP32 summation bound")
                error = float(np.max(np.abs(predictor.encode(direct)[0] - saved["memory__" + label])))
                if error > 1e-6:
                    raise ValueError("Independent decoded memory differs")
                zero = predictor.edit_raw(raw, codes, np.zeros(1024, dtype=np.float32)).reshape(-1)
                if not np.array_equal(zero, raw):
                    raise ValueError("Exported zero edit changed the source")
                direct_checks.append(dict(source=str(path), method=label, memory_error=error, fp32_bound=True, zero_exact=True))
    records = read_json(Path(config["parent_dictionary_run"]) / "descriptor_records.json")
    folds = training.crossfit.outer_folds(config)
    held, fit = folds[0], [v for v in config["train_video_ids"] if v not in folds[0]]
    assets = training.parent.load_assets(config, records, "sparse_edit", seed, 0, fit, held)
    part = training.design(assets, records, held, assets["reference"], True)
    gains = BoundedGains(1024, config["gain_bound"])
    state = torch.load(recovery / "training/inner_folds/sparse_edit" / f"seed{seed}/fold0/trainable_decoder/checkpoint.pt",
                       weights_only=True, map_location="cpu")
    gains.load_state_dict(state["gains"], strict=True)
    model = assets["model"]
    original_decoder = model.decoder.weight.detach().clone()
    model.load_state_dict(state["model"], strict=True)
    scale = torch.as_tensor(assets["scale"])
    with torch.no_grad():
        terms = training.loss_terms(model, gains, scale, original_decoder, part, config)
        edited = residual_edit(part["raw"], part["codes"], gains(), model, scale)
        scores = part["reference"](edited).numpy().astype(float) @ part["queries"].numpy().astype(float).T
        independent_tail = np.array([np.logaddexp(0, (np.quantile(scores[i, row["negatives"]], .99)
            - scores[i, row["positives"]]) / .05).mean() for i, row in enumerate(part["rows"])])
        tail_error = abs(float(terms[1]) - float(part["weights"].numpy() @ independent_tail))
    if tail_error > 1e-10:
        raise ValueError("Independent tail objective differs")
    atomic_write_json(recovery / "verification.json", dict(status="PASS", training_arrays_exact=count,
        application_arrays_exact=application_count, interruptions=interruptions, first_source_preserved=True,
        independent_tail_error=tail_error, direct_checks=direct_checks, source_sha256=evaluation.parent.digest(__file__)))
    print("DECODER_VERIFIED", count, application_count, tail_error, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    verify(args.run)
