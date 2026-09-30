import argparse
import json
import shutil
import time
from pathlib import Path

import discover_component_memory as base
import numpy as np

from src.acknowledgement_sae import file_sha256
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json


BASE_SOURCE_HASHES = base.source_hashes


def source_hashes():
    return dict(BASE_SOURCE_HASHES(), **{str(Path(__file__).relative_to(base.ROOT)): file_sha256(__file__)})


def parent_directory(config, context):
    relative = (Path("fit") / context["method"] / f"seed{context['seed']}" if context["fold"] == "full_training" else
                Path("inner_folds") / context["method"] / f"seed{context['seed']}" / f"fold{context['fold']}")
    return Path(config["parent_component_run"]) / relative


def verify_parent(config, context):
    directory = parent_directory(config, context)
    identity = read_json(directory / "identity.json")
    summary = read_json(directory / "summary.json")
    discovery = read_json(directory / "discovery.json")
    if summary["status"] != "COMPLETE" or summary["identity_sha256"] != base.shared.shared.json_digest(identity):
        raise ValueError("Parent component discovery identity is invalid")
    if identity["source_hashes"] != BASE_SOURCE_HASHES():
        raise ValueError("The parent component discovery implementation changed")
    for key in ("method", "seed", "fold"):
        if identity[key] != context[key]:
            raise ValueError("Parent component job scope differs")
    effects_hash = file_sha256(directory / "component_effects.npz")
    if (discovery["status"] != "COMPLETE" or discovery["effects_sha256"] != effects_hash
            or summary["discovery"]["effects_sha256"] != effects_hash):
        raise ValueError("Parent component effects differ from their receipts")
    receipt = dict(directory=str(directory.resolve()), effects_sha256=effects_hash,
                   discovery_sha256=file_sha256(directory / "discovery.json"),
                   identity_sha256=file_sha256(directory / "identity.json"))
    return directory, identity, discovery, receipt


def prepare_parent_receipts(config, phase, method, resume):
    parent = Path(config["parent_component_run"])
    parent_config = read_json(parent / "config.json")
    for key in ("parent_dictionary_run", "parent_head_run", "train_video_ids", "validation_video_ids",
                "reference_fit", "methods", "seeds", "component_counts", "intervention_strengths", "retention_floor"):
        if config[key] != parent_config[key]:
            raise ValueError(f"Reusable-component comparison changed a fixed setting: {key}")
    methods = config["methods"] if method == "all" else [method]
    seeds = config["seeds"][:1] if phase == "smoke" else config["seeds"]
    folds = [0] if phase == "smoke" else list(range(int(config["inner_folds"])))
    receipts = []
    for selected in methods:
        for seed in seeds:
            for fold in [*folds, "full_training"]:
                _, _, _, receipt = verify_parent(config, dict(method=selected, seed=seed, fold=fold))
                receipts.append(dict(method=selected, seed=seed, fold=fold, **receipt))
    output = Path(config["run_dir"]) / "smoke" if phase == "smoke" else Path(config["run_dir"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"ranking_parent_inputs_{method}.json"
    content = dict(parent_config_sha256=file_sha256(parent / "config.json"), jobs=receipts)
    if path.exists() and (not resume or read_json(path) != content):
        raise ValueError("Saved parent ranking inputs changed or require --resume")
    atomic_write_json(path, content)


def discover(config, assets, records, videos, directory, identity_hash, context, resume, stop_after_components):
    parent, identity, original, parent_receipt = verify_parent(config, context)
    if identity["fit_video_ids"] != videos:
        raise ValueError("Component ranking would use a different fitting population")
    expected_held = (config["validation_video_ids"] if context["fold"] == "full_training" else
                     read_json(Path(config["parent_dictionary_run"]) / "grouped_folds.json")["inner_held_folds"][context["fold"]])
    if identity["held_video_ids"] != expected_held:
        raise ValueError("Component ranking held population differs")
    for key in ("parent_directory", "parent_assets", "head_directory", "head_assets"):
        if identity[key] != assets["identity"][key]:
            raise ValueError("Component ranking dictionary, pooling, or head identity differs")
    ranking_identity = dict(identity_sha256=identity_hash, parent=parent_receipt)
    receipt_path = directory / "ranking_identity.json"
    if receipt_path.exists() and (not resume or read_json(receipt_path) != ranking_identity):
        raise ValueError("Reusable component ranking identity changed")
    atomic_write_json(receipt_path, ranking_identity)
    discovery_path = directory / "discovery.json"
    if discovery_path.exists():
        if not resume:
            raise FileExistsError("Use --resume for the completed reusable ranking")
        result = read_json(discovery_path)
        if (result["status"] != "COMPLETE" or result["identity_sha256"] != identity_hash
                or result["effects_sha256"] != file_sha256(directory / "component_effects.npz")
                or file_sha256(directory / "parent_component_effects.npz") != parent_receipt["effects_sha256"]):
            raise ValueError("Saved reusable component ranking differs")
    else:
        started = time.perf_counter()
        with np.load(parent / "component_effects.npz", allow_pickle=False) as archive:
            data = {key: archive[key].copy() for key in archive.files}
        benefit, harm = data["benefit"], data["harm"]
        if data["videos"].tolist() != sorted(videos) or benefit.shape != harm.shape or benefit.shape[1] != 1024:
            raise ValueError("Parent effect matrix has a different fitting scope")
        deletion_benefit, deletion_harm = [], []
        for omitted in range(len(videos)):
            b = np.delete(benefit, omitted, axis=0)
            h = np.delete(harm, omitted, axis=0)
            if not np.isfinite(b).any(axis=0).all() or not np.isfinite(h).any(axis=0).all():
                raise ValueError("A procedure deletion leaves an undefined ranking endpoint")
            deletion_benefit.append(np.nanmean(b, axis=0))
            deletion_harm.append(np.nanmean(h, axis=0))
        deletion_benefit, deletion_harm = np.stack(deletion_benefit), np.stack(deletion_harm)
        deletion_objective = deletion_benefit - deletion_harm
        robust = deletion_objective.min(axis=0)
        minimum_benefit = deletion_benefit.min(axis=0)
        eligible = np.flatnonzero((robust > 0) & (minimum_benefit > 0))
        ordering = eligible[np.lexsort((eligible, -robust[eligible]))]
        data.update(mean_objective=data["objective"].copy(), mean_eligible_order=data["eligible_order"].copy(),
                    objective=robust, eligible_order=ordering, deletion_benefit=deletion_benefit,
                    deletion_harm=deletion_harm, deletion_objective=deletion_objective,
                    deleted_videos=data["videos"].copy(), minimum_deletion_benefit=minimum_benefit)
        shutil.copyfile(parent / "component_effects.npz", directory / "parent_component_effects.npz")
        base.shared.shared.save_npz(directory / "component_effects.npz", **data)
        atomic_write_json(directory / "discovery_sources.json", original["source_counts"])
        result = dict(status="COMPLETE", identity_sha256=identity_hash, components=1024,
            eligible_components=len(ordering), ordered_components=ordering.tolist(),
            elapsed_seconds=time.perf_counter() - started, macro_benefit=data["macro_benefit"].tolist(),
            macro_harm=data["macro_harm"].tolist(), objective=robust.tolist(),
            mean_objective=data["mean_objective"].tolist(), source_counts=original["source_counts"],
            ranking="Minimum procedure-deletion mean benefit minus mean harm; each endpoint retains its own defined-procedure denominator.",
            single_component_scans_reused=True, omitted_procedures=data["videos"].tolist(), parent=parent_receipt,
            effects_sha256=file_sha256(directory / "component_effects.npz"))
        atomic_write_json(discovery_path, result)
        base.shared.progress(context, directory, "procedure_deletion_ranking", len(videos), len(videos), result["elapsed_seconds"])
        pause_after_checkpoint(discovery_path)
        if stop_after_components is not None:
            base.shared.progress(context, directory, "procedure_deletion_ranking", len(videos), len(videos),
                                 result["elapsed_seconds"], status="PAUSED")
            raise SystemExit(75)
    design = base.source_design(records, videos, assets["raw"], assets["valid"], assets["reference"], config["device"])
    return result, design


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("smoke", "fit"), required=True)
    parser.add_argument("--method", choices=("all",) + base.METHODS, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-components", choices=(1,), type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    prepare_parent_receipts(config, args.phase, args.method, args.resume)
    base.discover = discover
    base.source_hashes = source_hashes
    result = base.run(config, args.phase, args.method, args.resume, args.stop_after_components)
    print(json.dumps({key: result[key] for key in ("status", "phase", "completed_jobs", "total_jobs", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
