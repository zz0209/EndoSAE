import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import evaluate_conditioned_component_policy as evaluation
import train_conditioned_component_policy as training
from src import conditional_component_policy as policy
from src.checkpoint_io import atomic_write_json, read_json


def verify(run):
    config = read_json(run / "config.json")
    recovery = run / "recovery"
    recovery.mkdir(exist_ok=True)
    local = dict(config, run_dir=str(recovery))
    atomic_write_json(recovery / "config.json", local)
    command = [sys.executable, str(ROOT / "scripts/train_conditioned_component_policy.py"),
               "--config", str(recovery / "config.json"), "--smoke"]
    with (recovery / "execution.log").open("w", encoding="utf-8") as stream:
        stopped = subprocess.run(command + ["--stop-after-jobs", "1"], stdout=stream, stderr=subprocess.STDOUT)
        if stopped.returncode != 75:
            raise ValueError("Real job-boundary stop did not return75")
        first = recovery / "smoke/inner_folds/sparse_edit" / f"seed{config['seeds'][0]}" / "fold0/summary.json"
        signature = training.parent.file_sha256(first)
        resumed = subprocess.run(command + ["--resume"], stdout=stream, stderr=subprocess.STDOUT)
        if resumed.returncode != 0 or training.parent.file_sha256(first) != signature:
            raise ValueError("Resume failed or rewrote a valid completed policy")
    compared, arrays = 0, 0
    for method in config["methods"]:
        for fold in (0, "full_training"):
            relative = training.relative_job(method, config["seeds"][0], fold)
            original, restored = run / "smoke" / relative, recovery / "smoke" / relative
            for path in original.glob("*.npz"):
                with np.load(path, allow_pickle=False) as left, np.load(restored / path.name, allow_pickle=False) as right:
                    if left.files != right.files:
                        raise ValueError("Recovery changed array keys")
                    for key in left.files:
                        if not np.array_equal(left[key], right[key], equal_nan=True):
                            raise ValueError(f"Recovery changed numerical array: {relative}/{path.name}/{key}")
                        arrays += 1
                compared += 1
            left, right = read_json(original / "policy.json"), read_json(restored / "policy.json")
            for key in ("reference_threshold", "threshold", "advantage", "fixed_action", "fitting_mean_advantage"):
                if left[key] != right[key]:
                    raise ValueError("Recovery changed fitted policy")
    direct = []
    folder = run / "smoke/evaluation" / f"seed{config['seeds'][0]}"
    settings = read_json(folder / "evaluation_settings.json")
    reference = evaluation.parent.reference_api.Representations(Path(config["reference_fit"]))
    records = read_json(Path(config["parent_dictionary_run"]) / "descriptor_records.json")
    for item in read_json(folder / "source_policies.json")["sources"]:
        if item["status"] != "COMPLETE":
            continue
        prefix = folder / "source_policies" / item["population"] / item["video"] / item["episode_id"]
        with np.load(prefix / "predictions.npz", allow_pickle=False) as saved:
            original = saved["original_raw"].copy()
            scores_directory = folder / item["population"] / item["video"] / "sources" / item["episode_id"]
            population = read_json(Path(settings[item["population"] + "_base"]) / "config.json")
            query_dir = Path(population["descriptors"]) / item["video"]
            positions = np.flatnonzero(np.load(query_dir / "available.npy"))[:32]
            raw = np.load(query_dir / "raw_mean.npy", mmap_mode="r")[positions].copy()
            query = reference(raw)["supcon_l2"]
            with np.load(scores_directory / "scores.npz", allow_pickle=False) as actual:
                for method in config["methods"]:
                    bundle = evaluation.load_model(config, run / "smoke", method, config["seeds"][0], records)
                    features = saved[method + "__features"]
                    action, advantages, thresholds = policy.choose(bundle["policy"], features)
                    memory = saved[method + "__candidate_memories"][int(action)]
                    cosine = np.clip(query.astype(float) @ memory.astype(float), -1., 1.)
                    expected = 0.5 + 0.25 * (cosine - thresholds[int(action)])
                    name = ("sae" if method == "sparse_edit" else "dense") + "_conditional_calibrated"
                    error = float(np.max(np.abs(expected - actual[name][positions])))
                    if error > 1e-7 or advantages[0] != 0:
                        raise ValueError("Direct source-conditioned score verification failed")
                    zero = saved[method + "__candidate_raw"][0]
                    if not np.array_equal(zero, original):
                        raise ValueError("Zero edit changed original raw representation")
                    direct.append(dict(population=item["population"], method=method, queries=len(positions),
                                       max_error=error, zero_edit_exact=True))
    result = dict(status="PASS", stopped_exit_code=stopped.returncode, resumed_exit_code=resumed.returncode,
        completed_job_unchanged=True, npz_files_exact=compared, arrays_exact=arrays, direct_source_scores=direct)
    atomic_write_json(recovery / "verification.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    verify(args.run)
