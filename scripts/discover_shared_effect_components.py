import argparse
import json
import shutil
import time
from pathlib import Path

import discover_component_memory as base
import numpy as np
import torch

from src.acknowledgement_sae import file_sha256
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.token_memory_edit import FrozenSupCon


BASE_SOURCE_HASHES = base.source_hashes
BASE_LOAD_ASSETS = base.load_assets
BASE_EVALUATE = base.evaluate_candidate
PARENTS = {"original": "parent_component_run", "crossfit": "parent_crossfit_component_run"}


def source_hashes():
    return dict(BASE_SOURCE_HASHES(), **{str(Path(__file__).relative_to(base.ROOT)): file_sha256(__file__)})


def verify_parent(config, label, method, seed, fold, fit, held, assets):
    relative = (Path("fit") / method / f"seed{seed}" if fold == "full_training" else
                Path("inner_folds") / method / f"seed{seed}" / f"fold{fold}")
    directory = Path(config[PARENTS[label]]) / relative
    identity = read_json(directory / "identity.json")
    summary = read_json(directory / "summary.json")
    discovery = read_json(directory / "discovery.json")
    if (summary["status"] != "COMPLETE" or discovery["status"] != "COMPLETE"
            or summary["identity_sha256"] != base.shared.shared.json_digest(identity)):
        raise ValueError("Parent component result is incomplete or has a different identity")
    if (identity["method"] != method or identity["seed"] != seed or identity["fold"] != fold
            or identity["fit_video_ids"] != fit or identity["held_video_ids"] != held):
        raise ValueError("Parent component procedure scope differs")
    for key in ("parent_directory", "parent_assets", "head_directory", "head_assets"):
        if identity[key] != assets["identity"][key]:
            raise ValueError("Parent dictionary, normalization, pooling, or deployment head differs")
    if identity["records_sha256"] != file_sha256(Path(config["parent_dictionary_run"]) / "descriptor_records.json"):
        raise ValueError("Parent chronological record order differs")
    for name, expected in identity["source_hashes"].items():
        if file_sha256(base.ROOT / name) != expected:
            raise ValueError(f"Parent source changed: {name}")
    effects_hash = file_sha256(directory / "component_effects.npz")
    if effects_hash != discovery["effects_sha256"] or effects_hash != summary["discovery"]["effects_sha256"]:
        raise ValueError("Parent component effects differ from their receipts")
    if label == "crossfit":
        for spec in identity["discovery_heads"]:
            if set(spec["fit_video_ids"]) & set(spec["held_video_ids"]):
                raise ValueError("Parent discovery head includes its scored procedures")
            if set(spec["scope_video_ids"]) & set(held):
                raise ValueError("Parent discovery head includes an outer-held procedure")
            for name, expected in spec["assets"].items():
                if file_sha256(Path(spec["directory"]) / name) != expected:
                    raise ValueError("Parent crossfit discovery head changed")
        if sorted(v for spec in identity["discovery_heads"] for v in spec["held_video_ids"]) != sorted(fit):
            raise ValueError("Parent discovery heads do not partition the fitting procedures")
    return dict(label=label, directory=str(directory.resolve()), effects_sha256=effects_hash,
        summary_sha256=file_sha256(directory / "summary.json"), discovery_sha256=file_sha256(directory / "discovery.json"),
        identity_sha256=file_sha256(directory / "identity.json"), identity=identity, discovery=discovery)


def load_assets(config, records, method, seed, fold, fit_videos, held_videos):
    assets = BASE_LOAD_ASSETS(config, records, method, seed, fold, fit_videos, held_videos)
    parents = {label: verify_parent(config, label, method, seed, fold, fit_videos, held_videos, assets) for label in PARENTS}
    if parents["original"]["discovery"]["source_counts"] != parents["crossfit"]["discovery"]["source_counts"]:
        raise ValueError("Parent component effects use different source populations or aggregation")
    assets["shared_effect_parents"] = parents
    assets["identity"]["shared_effect_parents"] = {
        label: {key: p[key] for key in ("label", "directory", "effects_sha256", "summary_sha256", "discovery_sha256", "identity_sha256")}
        for label, p in parents.items()}
    return assets


@torch.no_grad()
def discover(config, assets, records, videos, directory, identity_hash, context, resume, stop_after_components):
    parents = assets["shared_effect_parents"]
    receipt_path = directory / "discovery.json"
    if receipt_path.exists():
        if not resume:
            raise FileExistsError("Completed shared-effect ranking requires --resume")
        result = read_json(receipt_path)
        if (result["status"] != "COMPLETE" or result["identity_sha256"] != identity_hash
                or result["effects_sha256"] != file_sha256(directory / "component_effects.npz")):
            raise ValueError("Saved shared-effect ranking changed")
        for label, parent in parents.items():
            if file_sha256(directory / f"{label}_component_effects.npz") != parent["effects_sha256"]:
                raise ValueError("Saved parent effect copy changed")
    else:
        started = time.perf_counter()
        arrays = {}
        for label, parent in parents.items():
            source = Path(parent["directory"]) / "component_effects.npz"
            with np.load(source, allow_pickle=False) as archive:
                data = {key: archive[key].copy() for key in archive.files}
            if (not np.array_equal(data["components"], np.arange(1024))
                    or data["videos"].tolist() != sorted(videos)
                    or data["benefit"].shape != (len(videos), 1024)
                    or data["harm"].shape != data["benefit"].shape
                    or not np.isfinite(data["objective"]).all()):
                raise ValueError("Parent component or procedure coordinates differ")
            arrays[label] = data
            shutil.copyfile(source, directory / f"{label}_component_effects.npz")
        original, crossfit = arrays["original"], arrays["crossfit"]
        objective = np.minimum(original["objective"], crossfit["objective"])
        eligible = np.flatnonzero((original["objective"] > 0) & (crossfit["objective"] > 0)
            & (original["macro_benefit"] > 0) & (crossfit["macro_benefit"] > 0))
        ordering = eligible[np.lexsort((eligible, -objective[eligible]))]
        combined = dict(components=np.arange(1024), videos=np.array(sorted(videos)), objective=objective,
                        eligible_order=ordering)
        for label, data in arrays.items():
            combined.update({label + "_" + key: data[key] for key in
                ("benefit", "harm", "macro_benefit", "macro_harm", "objective", "eligible_order")})
        base.shared.shared.save_npz(directory / "component_effects.npz", **combined)
        result = dict(status="COMPLETE", identity_sha256=identity_hash, components=1024, eligible_components=len(ordering),
            ordered_components=ordering.tolist(), objective=objective.tolist(),
            original_objective=original["objective"].tolist(), crossfit_objective=crossfit["objective"].tolist(),
            original_macro_benefit=original["macro_benefit"].tolist(), original_macro_harm=original["macro_harm"].tolist(),
            crossfit_macro_benefit=crossfit["macro_benefit"].tolist(), crossfit_macro_harm=crossfit["macro_harm"].tolist(),
            source_counts=parents["original"]["discovery"]["source_counts"],
            parents=assets["identity"]["shared_effect_parents"], elapsed_seconds=time.perf_counter() - started,
            effects_sha256=file_sha256(directory / "component_effects.npz"), single_component_scans_reused=True,
            ranking="Minimum original-head and head-excluded net effect; positive benefit and net effect required in both readouts; component counts are caps.")
        atomic_write_json(directory / "discovery_sources.json", result["source_counts"])
        atomic_write_json(receipt_path, result)
        base.shared.progress(context, directory, "shared_effect_ranking", 1024, 1024, result["elapsed_seconds"])
        pause_after_checkpoint(receipt_path)
        if stop_after_components is not None:
            base.shared.progress(context, directory, "shared_effect_ranking", 1024, 1024, result["elapsed_seconds"], status="PAUSED")
            raise SystemExit(75)
    parts = []
    for spec in parents["crossfit"]["identity"]["discovery_heads"]:
        local_assets = dict(assets, reference=FrozenSupCon(spec["directory"]).to(config["device"]))
        design = base.source_design(records, spec["held_video_ids"], assets["raw"], assets["valid"], local_assets["reference"], config["device"])
        parts.append((local_assets, design))
    assets["shared_crossfit_parts"] = parts
    fit_design = base.source_design(records, videos, assets["raw"], assets["valid"], assets["reference"], config["device"])
    return result, fit_design


def evaluate_candidate(config, assets, records, videos, candidate, ordering, directory, design):
    result = BASE_EVALUATE(config, assets, records, videos, candidate, ordering, directory, design)
    gains = torch.zeros(1024, device=config["device"])
    gains[result["selected_components"]] = -candidate["strength"]
    combined = {prefix: {key: {} for key in ("benefit", "harm")}
                for prefix in ("canonical", "variant", "variant_minus_canonical")}
    for local_assets, local_design in assets["shared_crossfit_parts"]:
        local = base.group_diagnostics(gains, local_assets, local_design)
        for prefix in combined:
            for key in combined[prefix]:
                if set(combined[prefix][key]) & set(local[prefix][key]):
                    raise ValueError("A crossfit diagnostic procedure was included twice")
                combined[prefix][key].update(local[prefix][key])
    result["discovery_crossfit_mechanism"] = combined
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("smoke", "fit"), required=True)
    parser.add_argument("--method", choices=("all",) + base.METHODS, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-components", choices=(1,), type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    for field in PARENTS.values():
        parent_config = read_json(Path(config[field]) / "config.json")
        for key in ("parent_dictionary_run", "parent_head_run", "train_video_ids", "validation_video_ids", "reference_fit",
                    "methods", "seeds", "inner_folds", "fold_seed", "component_counts", "intervention_strengths", "retention_floor"):
            if config[key] != parent_config[key]:
                raise ValueError(f"Shared-effect comparison changed a fixed setting: {key}")
    base.source_hashes = source_hashes
    base.load_assets = load_assets
    base.discover = discover
    base.evaluate_candidate = evaluate_candidate
    result = base.run(config, args.phase, args.method, args.resume, args.stop_after_components)
    print(json.dumps({key: result[key] for key in ("status", "phase", "completed_jobs", "total_jobs", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
