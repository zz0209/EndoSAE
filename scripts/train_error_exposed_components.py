import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import train_acknowledgement_sae as shared
from train_shared_task_components import evaluate_policy, optimize, pair_design, pair_values, progress, save_weights
from src.checkpoint_io import atomic_write_json, read_json
from src.shared_component_editor import correction_loss, edit_scores


def transfer_splits(records, folds, valid):
    splits = []
    excluded = []
    for fold in folds:
        if fold["fold"] == "full":
            continue
        pairs = shared.chronological_pairs(records, fold["held"])
        defined = sorted(video for video in fold["held"]
            if len({row["same_identity"] for row in pairs if row["video_id"] == video}) == 2)
        excluded.extend(sorted(set(fold["held"]) - set(defined)))
        for video in defined:
            fitting = sorted(set(fold["held"]) - {video})
            if set(fold["fit"]) & (set(fitting) | {video}) or video in fitting:
                raise ValueError("Task-model or correction leakage")
            designs = {"fit": pair_design(records, valid, fitting, True, "cpu"),
                       "held": pair_design(records, valid, [video], False, "cpu")}
            if any(row["video_id"] == video for row in designs["fit"]["pairs"]):
                raise ValueError("Evaluation procedure entered correction fitting")
            splits.append(dict(fold=fold["fold"], target=video, fit=fitting,
                               head_fit=fold["fit"], head=fold["head"]))
    if len(splits) != 14 or len({row["target"] for row in splits}) != 14 or len(excluded) != 5:
        raise ValueError("Unexpected transfer cohort")
    return splits, excluded


@torch.no_grad()
def add_parent_scores(target, original, embedding, code, directions, designs, config, mode):
    with np.load(original / "gains.npz", allow_pickle=False) as archive:
        gains = torch.as_tensor(archive["gains"], device=embedding.device)
    for name, design in designs.items():
        scores, _, _ = edit_scores(*pair_values(embedding, code, design), directions, gains,
                                   config["edit_budget"], mode)
        with np.load(target / f"{name}.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        arrays["in_sample"] = scores.cpu().numpy()
        shared.save_npz(target / f"{name}.npz", **arrays)


def run_batch(run, smoke, resume, stop_step, output_override):
    config = read_json(run / "config.json")
    parent = Path(config["parent_run"])
    parent_identity = read_json(parent / "identity.json")
    parent_training = read_json(parent / "training_summary.json")
    if parent_training["status"] != "COMPLETE":
        raise ValueError("Parent component experiment is incomplete")
    output = output_override or (run / "smoke" if smoke else run)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records, folds = read_json(parent / "records.json"), read_json(parent / "folds.json")
    reference = parent / f"models/fold0/seed{config['seeds'][0]}/sparse/basis.npz"
    with np.load(reference, allow_pickle=False) as archive:
        valid = archive["valid"].copy()
    splits, excluded = transfer_splits(records, folds, valid)
    if smoke:
        splits = [next(row for row in splits if row["fold"] == outer) for outer in [0, 2]]
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    parent_rows = {(row["fold"], row["seed"], row["basis"], row["mode"]): row
                   for row in parent_training["outputs"]}
    identity = dict(config=shared.file_sha256(run / "config.json"), protocol=shared.file_sha256(run / "protocol.json"),
        source={name: shared.file_sha256(ROOT / name) for name in ["scripts/train_error_exposed_components.py",
            "scripts/train_shared_task_components.py", "src/shared_component_editor.py",
            "scripts/train_token_memory_edit.py", "scripts/train_acknowledgement_sae.py"]},
        parent_identity=parent_identity, parent_summary=shared.file_sha256(parent / "training_summary.json"),
        records_sha256=shared.json_digest(records), splits=splits, smoke=smoke,
        torch=str(torch.__version__), numpy=np.__version__, python=sys.version)
    if (output / "identity.json").exists() and read_json(output / "identity.json") != identity:
        raise ValueError("Transfer batch identity changed")
    atomic_write_json(output / "identity.json", identity)
    atomic_write_json(output / "records.json", records)
    atomic_write_json(output / "splits.json", dict(splits=splits, excluded_one_class=excluded))
    total = len(splits) * len(seeds) * len(config["basis_methods"]) * len(config["modes"])
    completed, started, outputs = 0, time.perf_counter(), []
    for split in splits:
        designs = {name: pair_design(records, valid, videos, augmented, config["device"])
            for name, videos, augmented in [("fit", split["fit"], True), ("held", [split["target"]], False)]}
        for seed in seeds:
            for basis in config["basis_methods"]:
                original_basis = parent / f"models/fold{split['fold']}/seed{seed}/{basis}"
                basis_hash = shared.file_sha256(original_basis / "basis.npz")
                with np.load(original_basis / "basis.npz", allow_pickle=False) as archive:
                    np.testing.assert_array_equal(valid, archive["valid"])
                    embedding, code, directions = [torch.as_tensor(archive[key].copy(), device=config["device"])
                                                   for key in ["embedding", "code", "directions"]]
                for mode in config["modes"]:
                    original = original_basis / mode
                    receipt = read_json(original / "complete.json")
                    if shared.file_sha256(original / "gains.npz") != receipt["gains_sha256"]:
                        raise ValueError("Parent policy changed")
                    parent_row = parent_rows[(split["fold"], seed, basis, mode)]
                    if parent_row["fit_videos"] != split["head_fit"]:
                        raise ValueError("Basis fitting scope differs")
                    threshold = receipt["reports"]["fit"]["before_threshold"]
                    target = output / "models" / f"fold{split['fold']}_{split['target']}" / f"seed{seed}" / basis / mode
                    target.mkdir(parents=True, exist_ok=True)
                    signature = shared.json_digest(dict(batch=identity, split=split, seed=seed, basis=basis,
                        mode=mode, basis_hash=basis_hash, parent_gains=receipt["gains_sha256"]))
                    gate = nn.ParameterDict({"gains": nn.Parameter(torch.zeros(directions.shape[1], device=config["device"]))})
                    context = dict(output=output, completed=completed, total=total, started=started, seed=seed,
                        label=f"fold{split['fold']} target{split['target']} {seed} {basis} {mode}")
                    if (target / "complete.json").exists():
                        saved = read_json(target / "complete.json")
                        if not resume or saved["identity_sha256"] != signature or shared.file_sha256(target / "gains.npz") != saved["gains_sha256"]:
                            raise ValueError("Completed transfer policy changed")
                    else:
                        values = pair_values(embedding, code, designs["fit"])
                        def objective():
                            scores, _, _ = edit_scores(*values, directions, gate["gains"], config["edit_budget"], mode)
                            return correction_loss(scores, designs["fit"]["labels"], designs["fit"]["weights"],
                                threshold, config["temperature"], gate["gains"], config["gain_penalty"])
                        fitting = optimize(gate, objective, target, config, signature,
                            config["smoke_steps"] if smoke else config["gate_steps"], config["gate_learning_rate"],
                            context, resume, stop_step, False)
                        save_weights(target / "gains.npz", gate)
                        reports = evaluate_policy(target, embedding, code, directions, gate["gains"],
                                                  designs, config, mode, seed)
                        add_parent_scores(target, original, embedding, code, directions, designs, config, mode)
                        saved = dict(status="COMPLETE", identity_sha256=signature, fit=fitting, reports=reports,
                            gains_sha256=shared.file_sha256(target / "gains.npz"), basis_sha256=basis_hash,
                            completed_at=shared.now())
                        atomic_write_json(target / "complete.json", saved)
                    outputs.append(dict(directory=str(target), basis_directory=str(original_basis), parent_policy=str(original),
                        basis=basis, mode=mode, seed=seed, fold=split["fold"], fit_videos=split["fit"],
                        held_videos=[split["target"]], head_fit_videos=split["head_fit"], original_threshold=threshold,
                        receipt=saved))
                    completed += 1
                    progress(output, completed, total, context["label"], saved["fit"]["steps"],
                             saved["fit"]["steps"], started)
    atomic_write_json(output / "training_summary.json", dict(status="COMPLETE", outputs=outputs,
        completed_at=shared.now(), jobs=completed, seconds=time.perf_counter() - started, identity=identity))
    atomic_write_json(output / "training_progress.json", dict(status="COMPLETE", completed_jobs=completed,
        total_jobs=total, updated_at=shared.now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_batch(args.run, args.smoke, args.resume, args.stop_after_step, args.output)
