import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_frozen_identity_supcon as head_shared
import train_token_memory_edit as token
from src.acknowledgement_sae import file_sha256
from src.checkpoint_io import atomic_write_json, read_json
from src.token_memory_edit import METHODS, FrozenSupCon, TokenDictionary


def source_hashes():
    paths = [Path(__file__), Path(head_shared.__file__), Path(token.__file__),
             ROOT / "src/token_memory_edit.py", ROOT / "src/region_identity_sae.py",
             ROOT / "scripts/train_acknowledgement_sae.py", ROOT / "src/acknowledgement_sae.py",
             ROOT / "src/checkpoint_io.py"]
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}


def prepare_inputs(config, output):
    parent = Path(config["parent_run"])
    parent_config = read_json(parent / "config.json")
    records = read_json(parent / "descriptor_records.json")
    folds = read_json(parent / "grouped_folds.json")["inner_held_folds"]
    unchanged = ("train_video_ids", "validation_video_ids", "methods", "seeds", "inner_folds", "fold_seed",
                 "input_dim", "latent_dim", "top_k", "gain_bound", "gain_steps", "gain_checkpoints",
                 "gain_learning_rate", "identity_weight", "temperature", "batch_episodes", "reconstruction_batch",
                 "retention_floor", "reference_fit")
    if any(config[key] != parent_config[key] for key in unchanged):
        raise ValueError("Crossfit comparison changed a non-head training parameter")
    expected = token.shared.make_folds(records, config["train_video_ids"], int(config["inner_folds"]), int(config["fold_seed"]))
    if folds != expected:
        raise ValueError("Parent procedure folds differ")
    identity = dict(source_hashes=source_hashes(), parent_run=str(parent.resolve()),
                    parent_config_sha256=file_sha256(parent / "config.json"),
                    parent_records_sha256=file_sha256(parent / "descriptor_records.json"),
                    parent_inputs_sha256=file_sha256(parent / "token_inputs.json"),
                    head_config_sha256=file_sha256(config["head_config"]))
    for name, value in [("training_config.json", config), ("descriptor_records.json", records),
                        ("token_inputs.json", read_json(parent / "token_inputs.json"))]:
        path = output / name
        if path.exists() and read_json(path) != value:
            raise ValueError(f"Saved input changed: {path}")
        atomic_write_json(path, value)
    atomic_write_json(output / "grouped_folds.json", dict(training_procedures=config["train_video_ids"],
        validation_procedures=config["validation_video_ids"], inner_held_folds=folds,
        reference_scope="Each held fold is excluded from its dictionary, normalization, SupCon head and gain fitting. Full-data deployment uses the original 19-procedure SupCon head.",
        selection="Per-fold common threshold at 99% mean other-pair retention; equal weight for all eligible held procedures; earliest step tie; zero is included."))
    return parent, records, folds, identity


def fit_head(config, records, raw, raw_identity, held, directory, fold, steps, identity, output, resume, stop_stage, stop_after_step):
    directory.mkdir(parents=True, exist_ok=True)
    fit_videos = [v for v in config["train_video_ids"] if v not in held]
    selected = np.array([i for i, row in enumerate(records) if row["video_id"] in fit_videos])
    local_records = [records[i] for i in selected]
    head_config = read_json(config["head_config"])
    head_config.update(seed=int(config["head_seed"]), steps=steps)
    if head_config["projection"] != [768, 256, 128]:
        raise ValueError("Frozen reference requires the original head architecture")
    job_identity = dict(**identity, configuration_sha256=token.shared.json_digest(config),
                        kind="crossfit_head", fold=fold, fit_videos=fit_videos, held_videos=held,
                        steps=steps, seed=int(config["head_seed"]), raw_input=raw_identity)
    key = token.shared.json_digest(job_identity)
    summary_path = directory / "summary.json"
    if summary_path.exists():
        result = read_json(summary_path)
        if not resume or result["identity_sha256"] != key or result["status"] != "COMPLETE":
            raise ValueError("Existing crossfit head cannot be reused")
        for name in ("model.npz", "normalization.npz"):
            if file_sha256(directory / name) != result["assets"][name]:
                raise ValueError("Crossfit head asset changed")
        return result
    if (directory / "identity.json").exists() and read_json(directory / "identity.json") != job_identity:
        raise ValueError("Crossfit head identity changed")
    atomic_write_json(directory / "identity.json", job_identity)
    atomic_write_json(directory / "config.json", head_config)
    atomic_write_json(directory / "records.json", local_records)
    scaler = StandardScaler().fit(raw[selected])
    token.shared.save_npz(directory / "normalization.npz", mean=scaler.mean_, scale=scaler.scale_,
                          var=scaler.var_, n_samples_seen=scaler.n_samples_seen_)
    x = torch.from_numpy(scaler.transform(raw[selected]).astype(np.float32))
    labels = {label: i for i, label in enumerate(sorted({row["lesion_id"] for row in local_records}))}
    y = torch.tensor([labels[row["lesion_id"]] for row in local_records])
    torch.manual_seed(int(config["head_seed"]))
    model = head_shared.make_model(head_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(head_config["learning_rate"]),
                                 weight_decay=float(head_config["weight_decay"]))
    rng = np.random.default_rng(int(config["head_seed"]))
    checkpoint = directory / "head_checkpoint.pt"
    state = token.restore_checkpoint(checkpoint, model, optimizer, rng, key, "cpu", resume)
    first, elapsed_before = (int(state["step"]) + 1, float(state["elapsed_seconds"])) if state else (1, 0.)
    history, sequence = (state["history"], state["sequence"]) if state else ([], [])
    context = dict(phase="heads", method="shared_head", fold=fold, seed=int(config["head_seed"]),
                   completed_jobs=fold, total_jobs=int(config["inner_folds"]),
                   completed=fold, total=int(config["inner_folds"]),
                   progress_path=str((output / "head_progress.json").resolve()))
    started = time.perf_counter()
    for step in range(first, steps + 1):
        indices = head_shared.sample_batch(local_records, rng, head_config)
        sequence.append(indices)
        embeddings = F.normalize(model(x[indices]), dim=1)
        loss = head_shared.supcon(embeddings, y[indices], float(head_config["temperature"]))
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite crossfit head loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Invalid crossfit head gradient")
        gradient = float(torch.nn.utils.clip_grad_norm_(model.parameters(), float(head_config["gradient_clip_norm"])))
        if gradient <= 0:
            raise ValueError("Crossfit head objective has no gradient")
        optimizer.step()
        history.append(dict(step=step, loss=float(loss.detach()), gradient_norm=gradient, batch_size=len(indices)))
        stopping = stop_stage == "head" and stop_after_step is not None and step >= stop_after_step
        if step % int(config["gain_checkpoint_every"]) == 0 or step == steps or stopping:
            elapsed = elapsed_before + time.perf_counter() - started
            token.save_checkpoint(checkpoint, model, optimizer, rng, step, history, sequence, key, elapsed)
            token.progress(context, directory, "head", step, steps, elapsed, history[-1], "PAUSED" if stopping else "RUNNING")
            if stopping:
                raise SystemExit(75)
    token.shared.save_npz(directory / "model.npz", **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
    atomic_write_json(directory / "history.json", history)
    atomic_write_json(directory / "sequence.json", sequence)
    with torch.no_grad():
        expected = F.normalize(model(x), dim=1).numpy()
        actual = FrozenSupCon(directory)(torch.from_numpy(raw[selected])).numpy()
    if not np.array_equal(expected, actual):
        raise ValueError("Exported crossfit head changed predictions")
    result = dict(status="COMPLETE", identity_sha256=key, fold=fold, steps=steps,
                  fixed_checkpoint=True, fit_video_ids=fit_videos, held_video_ids=held,
                  fit_descriptors=len(selected), model_reload_exact=True, device="cpu", threads=int(config["threads"]),
                  elapsed_seconds=elapsed_before + time.perf_counter() - started,
                  assets={name: file_sha256(directory / name) for name in ("model.npz", "normalization.npz", "sequence.json")})
    atomic_write_json(summary_path, result)
    return result


def gain_job(config, method, seed, records, fit_videos, held_videos, directory, parent_directory,
             head_directory, steps, checkpoints, identity, context, resume, stop_stage, stop_after_step):
    directory.mkdir(parents=True, exist_ok=True)
    parent_identity = read_json(parent_directory / "identity.json")
    dictionary = read_json(parent_directory / "dictionary_summary.json")
    pool = read_json(parent_directory / "pool_summary.json")
    if parent_identity["fit_videos"] != fit_videos or parent_identity["held_videos"] != held_videos:
        raise ValueError("Parent dictionary procedure scope differs")
    if dictionary["status"] != "COMPLETE" or pool["status"] != "COMPLETE":
        raise ValueError("Incomplete parent dictionary or pooled views")
    parent_assets = {"model.npz": dictionary["model_sha256"], "normalization.npz": dictionary["normalization_sha256"],
                     "pooled_views.npz": pool["sha256"]}
    for name, expected in parent_assets.items():
        if file_sha256(parent_directory / name) != expected:
            raise ValueError(f"Parent asset differs: {parent_directory / name}")
    head_assets = {name: file_sha256(head_directory / name) for name in ("model.npz", "normalization.npz")}
    job_identity = dict(**identity, configuration_sha256=token.shared.json_digest(config),
                        method=method, seed=seed, fit_videos=fit_videos, held_videos=held_videos,
                        gain_steps=steps, gain_checkpoints=checkpoints,
                        parent_directory=str(parent_directory.resolve()), parent_assets=parent_assets,
                        head_directory=str(head_directory.resolve()), head_assets=head_assets)
    key = token.shared.json_digest(job_identity)
    if (directory / "summary.json").exists():
        summary = read_json(directory / "summary.json")
        if not resume or summary["status"] != "COMPLETE" or summary["identity_sha256"] != key:
            raise ValueError("Existing crossfit gain job cannot be reused")
        if file_sha256(directory / "gains.npz") != summary["gains_sha256"]:
            raise ValueError("Saved gain asset changed")
        return summary
    if (directory / "identity.json").exists() and read_json(directory / "identity.json") != job_identity:
        raise ValueError("Crossfit gain job identity changed")
    atomic_write_json(directory / "identity.json", job_identity)
    spec = read_json(parent_directory / "model_config.json")
    spec["reference_fit"] = str(head_directory.resolve())
    atomic_write_json(directory / "model_config.json", spec)
    for name in ("model.npz", "normalization.npz", "dictionary_summary.json", "dictionary_sequence.json", "pool_summary.json"):
        shutil.copyfile(parent_directory / name, directory / name)
    atomic_write_json(directory / "parent_assets.json", dict(directory=str(parent_directory.resolve()), assets=parent_assets))
    model = TokenDictionary(spec).to(config["device"])
    with np.load(directory / "model.npz", allow_pickle=False) as data:
        model.load_state_dict({k: torch.from_numpy(data[k].copy()) for k in data.files}, strict=True)
    model.eval().requires_grad_(False)
    with np.load(directory / "normalization.npz", allow_pickle=False) as data:
        mean, scale = data["mean"].copy(), data["scale"].copy()
    with np.load(parent_directory / "pooled_views.npz", allow_pickle=False) as data:
        raw, codes, valid = data["raw"].copy(), data["mean_codes"].copy(), data["valid"].copy()
    local_config = dict(config, reference_fit=str(head_directory.resolve()))
    gate = token.gain_fit(local_config, model, mean, scale, raw, codes, valid, records, fit_videos, held_videos,
                          directory, steps, checkpoints, key, seed, context, resume, stop_stage, stop_after_step)
    parent_sequence = read_json(parent_directory / "gain_sequence.json")
    sequence = read_json(directory / "gain_sequence.json")
    common = min(len(parent_sequence), len(sequence))
    if sequence[:common] != parent_sequence[:common]:
        raise ValueError("Gain sampling differs from the parent comparison")
    summary = dict(status="COMPLETE", identity_sha256=key, method=method, seed=seed, dictionary=dictionary,
                   reused_dictionary=True, reused_pool=True, parent_directory=str(parent_directory.resolve()),
                   head_directory=str(head_directory.resolve()), head_assets=head_assets,
                   parent_sequence_common_steps=common, **gate, completed_at=token.shared.now(),
                   runtime=dict(python=sys.version, numpy=np.__version__, sklearn=sklearn.__version__,
                                torch=str(torch.__version__), device=config["device"], threads=int(config["threads"])))
    atomic_write_json(directory / "summary.json", summary)
    token.progress(context, directory, "complete", steps, steps, gate["elapsed_seconds"], status="COMPLETE")
    return summary


def run(config, phase, selected_method, resume, stop_stage, stop_after_step):
    torch.set_num_threads(int(config["threads"]))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = Path(config["run_dir"]) / "smoke" if phase == "smoke" else Path(config["run_dir"])
    output.mkdir(parents=True, exist_ok=True)
    parent, records, folds, identity = prepare_inputs(config, output)
    seeds = config["seeds"][:1] if phase == "smoke" else config["seeds"]
    methods = list(config["methods"]) if selected_method == "all" else [selected_method]
    active_folds = folds[:1] if phase == "smoke" else folds
    head_steps = int(config["smoke_head_steps"] if phase == "smoke" else config["head_steps"])
    if phase in ("heads", "smoke"):
        source = parent / "inner_folds" / config["methods"][0] / f"seed{config['seeds'][0]}" / "fold0" / "pooled_views.npz"
        source_receipt = read_json(source.parent / "pool_summary.json")
        source_hash = file_sha256(source)
        if source_receipt["status"] != "COMPLETE" or source_hash != source_receipt["sha256"]:
            raise ValueError("Parent head input differs from its pooled-view receipt")
        raw_identity = dict(path=str(source.resolve()), sha256=source_hash, array="raw[:,0]")
        with np.load(source, allow_pickle=False) as data:
            raw = data["raw"][:, 0].copy()
        heads = [fit_head(config, records, raw, raw_identity, held, output / "heads" / f"fold{fold}", fold,
                          head_steps, identity, output, resume, stop_stage, stop_after_step)
                 for fold, held in enumerate(active_folds)]
        atomic_write_json(output / "head_summary.json", dict(status="COMPLETE", outputs=heads, **identity))
        atomic_write_json(output / "head_progress.json", dict(status="COMPLETE", phase="heads",
            completed=len(heads), total=len(heads), completed_jobs=len(heads), total_jobs=len(heads),
            updated_at=token.shared.now()))
        if phase == "heads":
            atomic_write_json(output / "training_progress.json", dict(status="COMPLETE", phase=phase,
                method="shared_head", completed_jobs=len(heads), total_jobs=len(heads), updated_at=token.shared.now()))
            return dict(status="COMPLETE", phase=phase, completed_jobs=len(heads), total_jobs=len(heads))
    total_jobs, completed_jobs = len(methods) * len(seeds) * (len(active_folds) + 1), 0
    gain_steps = int(config["smoke_gain_steps"] if phase == "smoke" else config["gain_steps"])
    checkpoints = [0, gain_steps] if phase == "smoke" else list(config["gain_checkpoints"])
    summaries, selections = [], []
    for seed in seeds:
        for method in methods:
            fold_results = []
            for fold, held in enumerate(active_folds):
                head_directory = output / "heads" / f"fold{fold}"
                head_summary = read_json(head_directory / "summary.json")
                fit = [v for v in config["train_video_ids"] if v not in held]
                if (head_summary["steps"] != head_steps or head_summary["held_video_ids"] != held
                        or head_summary["fit_video_ids"] != fit):
                    raise ValueError("Crossfit head does not match the required fold")
                for name in ("model.npz", "normalization.npz"):
                    if file_sha256(head_directory / name) != head_summary["assets"][name]:
                        raise ValueError("Crossfit head changed after fitting")
                relative = Path("inner_folds") / method / f"seed{seed}" / f"fold{fold}"
                context = dict(phase=phase, method=method, seed=seed, fold=fold,
                    completed_jobs=completed_jobs, total_jobs=total_jobs,
                    progress_path=str((output / "training_progress.json").resolve()))
                fold_results.append(gain_job(config, method, seed, records, fit, held, output / relative,
                    parent / relative, head_directory, gain_steps, checkpoints, identity, context,
                    resume, stop_stage, stop_after_step))
                completed_jobs += 1
            candidates = []
            for step in checkpoints:
                reports = [next(r for r in result["evaluations"] if r["step"] == step) for result in fold_results]
                rows = [row for report in reports for row in report["protection_point"]["procedure_rows"]]
                if len({row["video"] for row in rows}) != len(rows):
                    raise ValueError("Held procedure is counted more than once")
                candidates.append(dict(step=step, recall=float(np.mean([r["recall"] for r in rows])),
                    procedure_rows=rows, fold_points=[r["protection_point"] for r in reports]))
            selected = max(candidates, key=lambda row: (row["recall"], -row["step"]))["step"]
            selections.append(dict(method=method, seed=seed, selected_gain_steps=selected, candidates=candidates))
            atomic_write_json(output / f"checkpoint_selection_{method}.json", [r for r in selections if r["method"] == method])
            full_steps = gain_steps if phase == "smoke" else selected
            relative = Path("fit") / method / f"seed{seed}"
            context = dict(phase=phase, method=method, seed=seed, fold="full_training",
                completed_jobs=completed_jobs, total_jobs=total_jobs,
                progress_path=str((output / "training_progress.json").resolve()))
            result = gain_job(config, method, seed, records, config["train_video_ids"], config["validation_video_ids"],
                output / relative, parent / relative, Path(config["reference_fit"]), full_steps,
                [0] if full_steps == 0 else [0, full_steps], identity, context, resume, stop_stage, stop_after_step)
            summaries.append(dict(method=method, seed=seed, fit_directory=str((output / relative).resolve()), summary=result))
            completed_jobs += 1
    if source_hashes() != identity["source_hashes"]:
        raise ValueError("Training source changed during execution")
    result = dict(status="COMPLETE", phase=phase, outputs=summaries, checkpoint_selection=selections,
                  completed_jobs=completed_jobs, total_jobs=total_jobs, **identity)
    atomic_write_json(output / "training_summary.json", result)
    for method in methods:
        atomic_write_json(output / f"training_summary_{method}.json", dict(result,
            outputs=[r for r in summaries if r["method"] == method],
            checkpoint_selection=[r for r in selections if r["method"] == method]))
    atomic_write_json(output / "training_progress.json", dict(status="COMPLETE", phase=phase, method=selected_method,
        completed_jobs=completed_jobs, total_jobs=total_jobs, updated_at=token.shared.now()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("heads", "fit", "smoke"), required=True)
    parser.add_argument("--method", choices=("all",) + METHODS, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-stage", choices=("head", "gain"))
    parser.add_argument("--stop-after-step", type=int)
    args = parser.parse_args()
    if (args.stop_stage is None) != (args.stop_after_step is None):
        raise ValueError("Specify both --stop-stage and --stop-after-step")
    result = run(read_json(args.config), args.phase, args.method, args.resume, args.stop_stage, args.stop_after_step)
    print(json.dumps({key: result[key] for key in ("status", "phase", "completed_jobs", "total_jobs")}))


if __name__ == "__main__":
    main()
