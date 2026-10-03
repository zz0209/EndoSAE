import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_decoder_identity_tail as training
from src.checkpoint_io import atomic_write_json, read_json
from src.token_memory_edit import FrozenSupCon, residual_edit
from src.conditional_component_policy import source_weights, supervised_targets


def prepare(run, parent):
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    config = read_json(parent / "config.json")
    config.update(run_id=run.name, run_dir=str(run), decoder_run=str(parent))
    atomic_write_json(run / "config.json", config)
    atomic_write_json(run / "protocol.json", dict(
        question="Does the learned intervention fail across procedures while retaining its training head, or mainly when transferred to the evaluation head?",
        hypothesis="The almost-zero fitted tail loss can reflect procedure memorization, head-specific edits, or both. Holding the trained intervention fixed and crossing procedure exposure with head role distinguishes these explanations.",
        comparison="For each of48 saved decoder fits, evaluate each of its three training heads on its assigned fitting partition and on every outer-held procedure, then evaluate the original evaluation head on the same held procedures. Seven cells per fit,336 cells total. Fixed and trainable decoder arms, both dictionaries, all three seeds and all four fitted scopes remain included.",
        endpoint="Equal-procedure changes in utility, recall and identity-tail softplus loss relative to zero edit within each exact head/procedure cell. Include only existing noncanonical region views, matching the historical held endpoint. Also report canonical views separately to describe the actual source representation. No fitting, selection or application threshold changes.",
        interpretation="A positive change on new procedures through training heads that disappears through the evaluation head implicates head transfer. Failure on new procedures through both head roles implicates procedure generalization. Training-partition improvement only is in-sample evidence. Evaluation heads differ in both fit sample and fit size, so a head-role difference alone cannot identify its unique cause.",
        budget="CPU single thread; existing small pooled arrays and safe NPZ models. No GPU, full-video inference, raw data changes or independent reserved data. Completed per-cell assets resume without recomputation.",
        source_sha256=training.parent.file_sha256(__file__),
        parent_summary_sha256=training.parent.file_sha256(parent / "training_summary.json")))


@torch.no_grad()
def cell(assets, model, gains, records, videos, reference, include_canonical):
    part = training.design(assets, records, videos, reference, include_canonical=True)
    selected = [i for i, row in enumerate(part["rows"]) if (row["view"] == 0) == include_canonical]
    if not selected:
        raise ValueError("No observations for the selected view")
    part["rows"] = [part["rows"][i] for i in selected]
    for key in ("raw", "codes"):
        part[key] = part[key][selected]
    edited = residual_edit(part["raw"], part["codes"], gains, model, torch.as_tensor(assets["scale"]))
    memory = reference(edited).numpy()
    original = reference(part["raw"]).numpy()
    queries = part["queries"].numpy()
    rows, arrays = [], {}
    for index in range(len(part["rows"])):
        row = part["rows"][index]
        thresholds, utilities, scores = supervised_targets(np.vstack([original[index], memory[index]]),
            queries, row["positives"], row["negatives"], .99, .05)
        tails = [float(np.logaddexp(0, (threshold - score[row["positives"]]) / .05).mean())
                 for threshold, score in zip(thresholds, scores)]
        recalls = [float(np.mean(score[row["positives"]] > threshold)) for threshold, score in zip(thresholds, scores)]
        rows.append(dict(row, zero_utility=float(utilities[0]), utility=float(utilities[1]),
                         zero_recall=recalls[0], recall=recalls[1], zero_tail=tails[0], tail=tails[1]))
        arrays[f"scores_{index}"] = scores[:, row["positives"] + row["negatives"]]
    metrics = ("zero_utility", "utility", "zero_recall", "recall", "zero_tail", "tail")
    weights = source_weights(rows)
    means = {metric: float(weights @ np.array([r[metric] for r in rows])) for metric in metrics}
    procedures = {}
    for video in sorted({r["video_id"] for r in rows}):
        local = [r for r in rows if r["video_id"] == video]
        local_weights = source_weights(local)
        procedures[video] = {metric: float(local_weights @ np.array([r[metric] for r in local])) for metric in metrics}
    arrays.update(original_memory=original, memory=memory, edited_raw=edited.numpy())
    return dict(means=means, procedures=procedures, rows=rows, excluded=part["excluded"]), arrays


def run_batch(run, output, smoke, resume, stop_after):
    config = read_json(run / "config.json")
    parent = Path(config["decoder_run"])
    jobs = read_json(parent / "training_summary.json")["outputs"]
    if smoke:
        jobs = [j for j in jobs if j["seed"] == config["seeds"][0] and j["fold"] in (0, "full_training")]
    records = read_json(Path(config["parent_dictionary_run"]) / "descriptor_records.json")
    output.mkdir(parents=True, exist_ok=True)
    results, started = [], time.perf_counter()
    signature = training.parent.file_sha256(__file__)
    dependencies = training.hashes()
    for job in jobs:
        folder = Path(job["folder"])
        identity = read_json(folder / "identity.json")
        for name in ("model.npz", "gains.npz"):
            if training.parent.file_sha256(folder / name) != job["artifacts"][name]:
                raise ValueError("Parent model or gains changed")
        assets = training.parent.load_assets(config, records, job["method"], job["seed"], job["fold"],
                                             identity["fit_videos"], identity["held_videos"])
        model = assets["model"]
        with np.load(folder / "model.npz", allow_pickle=False) as data:
            model.load_state_dict({k: torch.from_numpy(data[k].copy()) for k in data.files}, strict=True)
        with np.load(folder / "gains.npz", allow_pickle=False) as data:
            gains = torch.from_numpy(data["gains"].copy())
        plans = [training.crossfit.verify_head(config, p) for p in training.crossfit.head_plan(config, records, job["fold"])]
        cells = []
        for index, plan in enumerate(plans):
            if set(plan["scope_video_ids"]) & set(identity["held_videos"]):
                raise ValueError("Held procedures were exposed to a training head")
            reference = FrozenSupCon(plan["directory"])
            cells.extend([(f"training{index}", "fit", plan["held_video_ids"], reference),
                          (f"training{index}", "held", identity["held_videos"], reference)])
        cells.append(("evaluation", "held", identity["held_videos"], assets["reference"]))
        for head, cohort, videos, reference in cells:
            name = f"{job['method']}_seed{job['seed']}_fold{job['fold']}_{job['arm']}_{head}_{cohort}"
            target = output / "cells" / name
            target.mkdir(parents=True, exist_ok=True)
            stamp = dict(source_sha256=signature, dependencies=dependencies,
                         config_sha256=training.parent.file_sha256(run / "config.json"),
                         parent_job_sha256=training.parent.file_sha256(folder / "summary.json"),
                         method=job["method"], seed=job["seed"], fold=job["fold"], arm=job["arm"],
                         head=head, cohort=cohort, videos=videos, plans=plans)
            key = training.parent.shared.shared.json_digest(stamp)
            if (target / "summary.json").exists():
                report = read_json(target / "summary.json")
                if not resume or report["identity_sha256"] != key:
                    raise ValueError("Cell identity changed or resume is absent")
                for name, digest in report["artifacts"].items():
                    if training.parent.file_sha256(target / name) != digest:
                        raise ValueError("Saved cell changed")
            else:
                reports = {}
                for canonical in (False, True):
                    view = "canonical" if canonical else "region"
                    report, arrays = cell(assets, model, gains, records, videos, reference, canonical)
                    atomic_write_json(target / (view + ".json"), report)
                    training.parent.shared.shared.save_npz(target / (view + ".npz"), **arrays)
                    reports[view] = report
                errors = {}
                if head == "evaluation":
                    for metric in job["held"]:
                        errors[metric] = abs(reports["region"]["means"][metric] - job["held"][metric])
                    if max(errors.values()) > 1e-12:
                        raise ValueError("Recomputed historical held endpoint differs")
                report = dict(**stamp, identity_sha256=key, views=reports, historical_errors=errors,
                              artifacts={p.name: training.parent.file_sha256(p) for p in target.iterdir() if p.suffix in (".npz", ".json")})
                atomic_write_json(target / "summary.json", report)
            results.append(report)
            elapsed = time.perf_counter() - started
            progress = dict(status="RUNNING", completed=len(results), total=len(jobs) * 7, elapsed_seconds=elapsed,
                            updated_at=training.parent.shared.shared.now(), last=name,
                            remaining_seconds=(len(jobs) * 7 - len(results)) * elapsed / len(results))
            atomic_write_json(output / "progress.json", progress)
            print("HEAD_TRANSFER", progress, flush=True)
            if stop_after and len(results) >= stop_after:
                raise SystemExit(75)
    if signature != training.parent.file_sha256(__file__):
        raise ValueError("Source changed during evaluation")
    atomic_write_json(output / "summary.json", dict(status="COMPLETE", cells=results, elapsed_seconds=time.perf_counter() - started))
    atomic_write_json(output / "progress.json", dict(progress, status="COMPLETE"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.prepare:
        prepare(args.run, args.prepare)
    else:
        run_batch(args.run, args.output or (args.run / "smoke" if args.smoke else args.run), args.smoke, args.resume, args.stop_after)
