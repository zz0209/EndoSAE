import argparse
import json
import time
from pathlib import Path

import discover_component_memory as base
import train_crossfit_memory_edit as head_training
import numpy as np
import torch

from src.acknowledgement_sae import file_sha256
from src.checkpoint_io import atomic_write_json, pause_after_checkpoint, read_json
from src.token_memory_edit import FrozenSupCon


BASE_SOURCE_HASHES = base.source_hashes
BASE_LOAD_ASSETS = base.load_assets
BASE_DISCOVER = base.discover
BASE_EVALUATE = base.evaluate_candidate
BASE_PROGRESS = base.shared.progress


def source_hashes():
    result = dict(BASE_SOURCE_HASHES(), **head_training.source_hashes())
    result[str(Path(__file__).relative_to(base.ROOT))] = file_sha256(__file__)
    return result


def outer_folds(config):
    return read_json(Path(config["parent_dictionary_run"]) / "grouped_folds.json")["inner_held_folds"]


def head_plan(config, records, fold):
    outer = outer_folds(config)
    if fold == "full_training":
        scope, partitions = config["train_video_ids"], outer
    else:
        scope = [v for v in config["train_video_ids"] if v not in outer[fold]]
        partitions = base.shared.shared.make_folds(records, scope, int(config["discovery_head_folds"]), int(config["fold_seed"]))
    plans = []
    for inner, held in enumerate(partitions):
        directory = (Path(config["parent_head_run"]) / "heads" / f"fold{inner}" if fold == "full_training" else
                     Path(config.get("discovery_heads_root", Path(config["run_dir"]) / "discovery_heads")) / f"outer{fold}" / f"inner{inner}")
        plans.append(dict(outer_fold=fold, inner_fold=inner, directory=str(directory.resolve()),
            fit_video_ids=[v for v in scope if v not in held], held_video_ids=held, scope_video_ids=scope))
    if sorted(v for p in plans for v in p["held_video_ids"]) != sorted(scope):
        raise ValueError("Discovery procedure partition is not unique and complete")
    return plans


def verify_head(config, plan):
    directory = Path(plan["directory"])
    summary = read_json(directory / "summary.json")
    identity = read_json(directory / "identity.json")
    if (summary["status"] != "COMPLETE" or summary["steps"] != int(config["head_steps"])
            or identity["seed"] != int(config["head_seed"])
            or summary["fit_video_ids"] != plan["fit_video_ids"]
            or summary["held_video_ids"] != plan["held_video_ids"]
            or set(plan["fit_video_ids"]) & set(plan["held_video_ids"])
            or summary["identity_sha256"] != base.shared.shared.json_digest(identity)):
        raise ValueError("Discovery head exposure or identity differs")
    assets = {name: file_sha256(directory / name) for name in ("model.npz", "normalization.npz")}
    if any(assets[name] != summary["assets"][name] for name in assets):
        raise ValueError("Discovery head asset changed")
    return dict(plan, assets=assets, identity_sha256=summary["identity_sha256"],
                summary_sha256=file_sha256(directory / "summary.json"))


def heads(config, smoke, resume, stop_after_step):
    torch.set_num_threads(int(config["threads"]))
    torch.use_deterministic_algorithms(True)
    output = Path(config["run_dir"])
    output.mkdir(parents=True, exist_ok=True)
    parent = Path(config["parent_dictionary_run"])
    records = read_json(parent / "descriptor_records.json")
    raw_path = parent / "inner_folds" / config["methods"][0] / f"seed{config['seeds'][0]}" / "fold0" / "pooled_views.npz"
    receipt = read_json(raw_path.parent / "pool_summary.json")
    raw_hash = file_sha256(raw_path)
    if receipt["status"] != "COMPLETE" or receipt["sha256"] != raw_hash:
        raise ValueError("Head input differs from the pooled-view receipt")
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = archive["raw"][:, 0].copy()
    raw_identity = dict(path=str(raw_path.resolve()), sha256=raw_hash, array="raw[:,0]")
    identity = dict(source_hashes=source_hashes(), records_sha256=file_sha256(parent / "descriptor_records.json"),
                    head_config_sha256=file_sha256(config["head_config"]))
    plans = [p for fold in ([0] if smoke else range(3)) for p in head_plan(config, records, fold)]
    outputs, started = [], time.perf_counter()
    progress_state = dict(completed_jobs=0, total_jobs=len(plans), completed=0, total=len(plans))

    def progress(context, directory, stage, step, steps, elapsed, losses=None, status="RUNNING"):
        BASE_PROGRESS(dict(context, **progress_state), directory, stage, step, steps, elapsed, losses, status)

    head_training.token.progress = progress
    for plan in plans:
        local_config = dict(train_video_ids=plan["scope_video_ids"], head_config=config["head_config"],
            head_seed=int(config["head_seed"]), head_steps=int(config["head_steps"]),
            gain_checkpoint_every=int(config["gain_checkpoint_every"]),
            inner_folds=int(config["discovery_head_folds"]), threads=int(config["threads"]))
        head_identity = dict(identity, outer_fold=plan["outer_fold"], inner_fold=plan["inner_fold"],
            head_exposure="Head fitting and normalization exclude this discovery partition and every outer-held procedure.")
        result = head_training.fit_head(local_config, records, raw, raw_identity, plan["held_video_ids"],
            Path(plan["directory"]), plan["inner_fold"], int(config["head_steps"]), head_identity, output,
            resume, "head" if stop_after_step is not None else None, stop_after_step)
        outputs.append(dict(**plan, summary=result))
        progress_state.update(completed_jobs=len(outputs), completed=len(outputs))
        status = "COMPLETE" if len(outputs) == len(plans) else "RUNNING"
        if source_hashes() != identity["source_hashes"]:
            raise ValueError("Discovery head training source changed during execution")
        atomic_write_json(output / "head_summary.json", dict(status=status, smoke_heads=smoke,
            selected_outer_folds=[0] if smoke else [0, 1, 2], completed_jobs=len(outputs), total_jobs=len(plans),
            outputs=outputs, **identity))
        atomic_write_json(output / "head_progress.json", dict(status=status, phase="heads", method="shared_head",
            stage="head", step=int(config["head_steps"]), steps=int(config["head_steps"]),
            **progress_state, elapsed_seconds=time.perf_counter() - started, updated_at=base.shared.shared.now()))
    head_training.token.progress = BASE_PROGRESS
    return dict(status="COMPLETE", phase="heads", completed_jobs=len(outputs), total_jobs=len(plans),
                elapsed_seconds=time.perf_counter() - started)


def load_assets(config, records, method, seed, fold, fit_videos, held_videos):
    assets = BASE_LOAD_ASSETS(config, records, method, seed, fold, fit_videos, held_videos)
    specs = [verify_head(config, p) for p in head_plan(config, records, fold)]
    if sorted(v for p in specs for v in p["held_video_ids"]) != sorted(fit_videos):
        raise ValueError("Discovery head partitions differ from the dictionary fitting scope")
    if any(set(p["scope_video_ids"]) & set(held_videos) for p in specs):
        raise ValueError("Outer held data entered discovery head fitting or scoring")
    assets["discovery_heads"] = specs
    assets["identity"]["discovery_heads"] = specs
    return assets


def baseline(design):
    scores = design["before"].detach().cpu().numpy()
    result = {}
    for key in ("positive", "negative"):
        values = scores[design[key].cpu().numpy()]
        result[key] = dict(pairs=len(values), mean=float(values.mean()) if len(values) else None,
            quantiles=np.quantile(values, [0, .1, .5, .9, 1]).tolist() if len(values) else None)
    result["source_counts"] = design["metadata"]
    result["unit"] = "Descriptive source-view/query score distribution; procedure remains the independent unit."
    return result


@torch.no_grad()
def discover(config, assets, records, videos, directory, identity_hash, context, resume, stop_after_components):
    parts, scans, all_videos = [], [], sorted(videos)
    benefits = np.full((len(videos), 1024), np.nan)
    harms = np.full_like(benefits, np.nan)
    assigned = []
    started = time.perf_counter()
    for spec in assets["discovery_heads"]:
        inner = spec["inner_fold"]
        local_directory = directory / "crossfit_discovery" / f"inner{inner}"
        local_directory.mkdir(parents=True, exist_ok=True)
        local_identity = dict(job_identity_sha256=identity_hash, discovery_head=spec,
                              dictionary=assets["identity"]["parent_assets"], scored_videos=spec["held_video_ids"])
        key = base.shared.shared.json_digest(local_identity)
        identity_path = local_directory / "identity.json"
        if identity_path.exists() and (not resume or read_json(identity_path) != local_identity):
            raise ValueError("Saved discovery partition identity changed or requires --resume")
        atomic_write_json(identity_path, local_identity)
        local_assets = dict(assets, reference=FrozenSupCon(spec["directory"]).to(config["device"]))
        local_context = dict(context, discovery_inner_fold=inner, discovery_inner_folds=len(assets["discovery_heads"]))
        discovery_path = local_directory / "discovery.json"
        if discovery_path.exists():
            scan = read_json(discovery_path)
            if scan["status"] != "COMPLETE" or scan["effects_sha256"] != file_sha256(local_directory / "component_effects.npz"):
                raise ValueError("Saved discovery partition effects changed")
            design = base.source_design(records, spec["held_video_ids"], assets["raw"], assets["valid"],
                                        local_assets["reference"], config["device"])
        else:
            scan, design = BASE_DISCOVER(config, local_assets, records, spec["held_video_ids"], local_directory,
                key, local_context, resume, stop_after_components)
        baseline_path = local_directory / "baseline_pairs.json"
        baseline_receipt = local_directory / "baseline.json"
        if baseline_path.exists() and baseline_receipt.exists():
            saved_baseline = read_json(baseline_receipt)
            if not resume or saved_baseline["pairs_sha256"] != file_sha256(baseline_path):
                raise ValueError("Saved discovery head baseline changed or requires --resume")
        else:
            zero = torch.zeros(1024, device=config["device"])
            report = base.shared.evaluate_gains(config, assets["model"], lambda: zero, local_assets["reference"],
                torch.from_numpy(assets["raw"]).to(config["device"]),
                torch.from_numpy(assets["codes"]).to(config["device"]), assets["valid"],
                torch.from_numpy(assets["scale"]).to(config["device"]), records, spec["held_video_ids"],
                local_directory, "baseline_pairs")
            atomic_write_json(baseline_receipt, dict(baseline(design),
                protection_point=report["protection_point"], macro_procedure_auroc=report["macro_procedure_auroc"],
                pairs_sha256=file_sha256(baseline_path), used_for_selection=False))
        with np.load(local_directory / "component_effects.npz", allow_pickle=False) as archive:
            local_videos = archive["videos"].tolist()
            if local_videos != sorted(spec["held_video_ids"]):
                raise ValueError("Saved component effects have a different discovery procedure scope")
            for index, video in enumerate(local_videos):
                if video in assigned:
                    raise ValueError("A discovery procedure was scored more than once")
                destination = all_videos.index(video)
                benefits[destination], harms[destination] = archive["benefit"][index], archive["harm"][index]
                assigned.append(video)
        parts.append((local_assets, design))
        scans.append(dict(inner_fold=inner, head=spec, effect_sha256=scan["effects_sha256"],
                          source_counts=scan["source_counts"], elapsed_seconds=scan["elapsed_seconds"]))
    if sorted(assigned) != all_videos:
        raise ValueError("Some fitting procedures lack crossfit discovery effects")
    benefit, harm = np.nanmean(benefits, axis=0), np.nanmean(harms, axis=0)
    objective = benefit - harm
    if not np.isfinite(objective).all():
        raise ValueError("Crossfit component aggregate is undefined")
    eligible = np.flatnonzero((objective > 0) & (benefit > 0))
    ordering = eligible[np.lexsort((eligible, -objective[eligible]))]
    base.shared.shared.save_npz(directory / "component_effects.npz", components=np.arange(1024), videos=np.array(all_videos),
        benefit=benefits, harm=harms, macro_benefit=benefit, macro_harm=harm, objective=objective, eligible_order=ordering)
    source_counts = dict(parts[0][1]["metadata"])
    source_counts.update(videos=all_videos,
        **{key: sum(p[1]["metadata"][key] for p in parts) for key in ("sources", "source_views", "sources_with_both")},
        by_video={video: values for p in parts for video, values in p[1]["metadata"]["by_video"].items()})
    result = dict(status="COMPLETE", identity_sha256=identity_hash, components=1024, eligible_components=len(ordering),
        ordered_components=ordering.tolist(), macro_benefit=benefit.tolist(), macro_harm=harm.tolist(),
        objective=objective.tolist(), source_counts=source_counts, discovery_heads=scans,
        elapsed_seconds=sum(s["elapsed_seconds"] for s in scans), assembly_seconds=time.perf_counter() - started,
        effects_sha256=file_sha256(directory / "component_effects.npz"),
        discovery_scope="Each procedure is scored once through a head whose fitting and normalization excluded it.",
        fit_mechanism_scope="Candidate fit_mechanism uses the original outer-fit or full19 deployment head.")
    atomic_write_json(directory / "discovery.json", result)
    atomic_write_json(directory / "discovery_sources.json", source_counts)
    assets["crossfit_discovery_parts"] = parts
    pause_after_checkpoint(directory / "discovery.json")
    return result, base.source_design(records, videos, assets["raw"], assets["valid"], assets["reference"], config["device"])


def evaluate_candidate(config, assets, records, videos, candidate, ordering, directory, design):
    result = BASE_EVALUATE(config, assets, records, videos, candidate, ordering, directory, design)
    gains = torch.zeros(1024, device=config["device"])
    gains[result["selected_components"]] = -candidate["strength"]
    combined = {prefix: {key: {} for key in ("benefit", "harm")}
                for prefix in ("canonical", "variant", "variant_minus_canonical")}
    for local_assets, local_design in assets["crossfit_discovery_parts"]:
        local = base.group_diagnostics(gains, local_assets, local_design)
        for prefix in combined:
            for key in combined[prefix]:
                if set(combined[prefix][key]) & set(local[prefix][key]):
                    raise ValueError("Crossfit group diagnostics repeat a procedure")
                combined[prefix][key].update(local[prefix][key])
    result["discovery_crossfit_mechanism"] = combined
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=("heads", "smoke", "fit"), required=True)
    parser.add_argument("--method", choices=("all",) + base.METHODS, default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-heads", action="store_true")
    parser.add_argument("--stop-after-head-step", type=int)
    parser.add_argument("--stop-after-components", type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    if int(config["discovery_head_folds"]) != 3 or int(config["head_steps"]) != 750:
        raise ValueError("The fixed discovery-head budget differs")
    if args.phase == "heads":
        result = heads(config, args.smoke_heads, args.resume, args.stop_after_head_step)
    else:
        if args.smoke_heads or args.stop_after_head_step is not None:
            raise ValueError("Head-only options require the heads phase")
        base.source_hashes = source_hashes
        base.load_assets = load_assets
        base.discover = discover
        base.evaluate_candidate = evaluate_candidate
        result = base.run(config, args.phase, args.method, args.resume, args.stop_after_components)
    print(json.dumps({key: result[key] for key in ("status", "phase", "completed_jobs", "total_jobs", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
