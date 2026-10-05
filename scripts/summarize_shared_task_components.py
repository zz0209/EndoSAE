import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_io import atomic_write_json, read_json


def proportion(values, mask):
    return float(np.mean(values[mask])) if np.any(mask) else None


def threshold(scores, labels, weights, quantile):
    selected = (~labels) & (weights > 0)
    return float(np.quantile(scores[selected], quantile, method="inverted_cdf", weights=weights[selected]))


def summarize(run, directory):
    config = read_json(run / "config.json")
    roster = read_json(directory / "records.json")
    training = read_json(directory / "training_summary.json")
    if training["status"] != "COMPLETE":
        raise ValueError("Training is incomplete")
    rows, components, representations = [], [], []
    for item in training["outputs"]:
        folder = Path(item["directory"])
        with np.load(folder / "fit.npz", allow_pickle=False) as fit:
            original_threshold = (item["original_threshold"] if "original_threshold" in item else
                threshold(fit["before"], fit["labels"], fit["weights"], config["negative_quantile"]))
            fitting_thresholds = {name: threshold(fit[name], fit["labels"], fit["weights"], config["negative_quantile"])
                                  for name in ["before", "after"] + config.get("comparison_variants", []) +
                                  [f"random{i}" for i in range(config["random_controls"]) ]}
        for partition in ["fit", "held"]:
            pairs = read_json(folder / f"{partition}_pairs.json")
            with np.load(folder / f"{partition}.npz", allow_pickle=False) as archive:
                values = {key: archive[key].copy() for key in archive.files}
            labels = values["labels"]
            videos = np.array([p["video_id"] for p in pairs])
            cross = labels & ~np.array([p["same_annotation_interval"] for p in pairs])
            before = values["before"] > original_threshold
            for variant in fitting_thresholds:
                scores = values[variant]
                boundaries = dict(original_fit=original_threshold, corrected_fit=fitting_thresholds[variant],
                    descriptive_held=threshold(scores, labels, values["weights"], config["negative_quantile"]))
                for boundary, value in boundaries.items():
                    decisions = scores > value
                    for video in sorted(set(videos)):
                        selected = videos == video
                        positive, negative = selected & labels, selected & ~labels
                        if not positive.any() or not negative.any():
                            continue
                        true_positive, false_positive = positive & before, negative & before
                        true_negative, false_negative = negative & ~before, positive & ~before
                        rows.append(dict(basis=item["basis"], mode=item["mode"], seed=item["seed"], fold=item["fold"],
                            partition=partition, video_id=video, variant=variant, boundary=boundary, threshold=value,
                            auroc=float(roc_auc_score(labels[selected], scores[selected])),
                            recall=proportion(decisions, positive), negative_retention=proportion(~decisions, negative),
                            cross_interval_recall=proportion(decisions, selected & cross),
                            baseline_false_positive=int(false_positive.sum()), baseline_true_positive=int(true_positive.sum()),
                            corrected_false_positive=int(np.sum(false_positive & ~decisions)),
                            damaged_true_positive=int(np.sum(true_positive & ~decisions)),
                            introduced_false_positive=int(np.sum(true_negative & decisions)), recovered_positive=int(np.sum(false_negative & decisions)),
                            false_positive_correction=proportion(~decisions, false_positive),
                            true_positive_damage=proportion(~decisions, true_positive),
                            negative_score_change=proportion(scores - values["before"], negative),
                            positive_score_change=proportion(scores - values["before"], positive),
                            mean_edit_norm=float(values["edit_norm"][selected].mean()),
                            mean_shared_active=float(values["shared_active"][selected].mean()), pairs=int(selected.sum())))
            for index, component in enumerate(values["component_indices"]):
                effect = values["component_effects"][index]
                for video in sorted(set(videos)):
                    mask = videos == video
                    components.append(dict(basis=item["basis"], mode=item["mode"], seed=item["seed"], fold=item["fold"],
                        partition=partition, video_id=video, component=int(component),
                        positive_effect=proportion(effect, mask & labels), negative_effect=proportion(effect, mask & ~labels)))
        if item["mode"] == "bilateral":
            with np.load(Path(item.get("basis_directory", folder.parent)) / "basis.npz", allow_pickle=False) as basis:
                for video in item["held_videos"]:
                    mask = np.array([row["video_id"] == video for row in roster])
                    valid = basis["valid"][mask]
                    representations.append(dict(basis=item["basis"], seed=item["seed"], fold=item["fold"], video_id=video,
                        nmse=float(basis["nmse"][mask][valid].mean()),
                        active=float((basis["code"][mask][valid] != 0).sum(-1).mean())))
    output = directory / "analysis"
    output.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    component_table = pd.DataFrame(components)
    representation_table = pd.DataFrame(representations)
    table.to_csv(output / "procedures.csv", index=False)
    component_table.to_csv(output / "component_effects.csv", index=False)
    representation_table.to_csv(output / "representations.csv", index=False)
    table["scope"] = np.where(table.fold == "full", "examined_validation", "training_held")
    metrics = ["recall", "negative_retention", "auroc", "cross_interval_recall", "false_positive_correction",
               "true_positive_damage", "positive_score_change", "negative_score_change", "mean_edit_norm", "mean_shared_active"]
    seeds = table[table.partition == "held"].groupby(["scope", "basis", "mode", "variant", "boundary", "seed"])[metrics].mean().reset_index()
    aggregate = seeds.groupby(["scope", "basis", "mode", "variant", "boundary"])[metrics].mean().reset_index()
    seeds.to_csv(output / "seeds.csv", index=False)
    aggregate.to_csv(output / "summary.csv", index=False)
    shown_variants = ["before"] + config.get("comparison_variants", []) + ["after"]
    policy = aggregate[(aggregate.variant.isin(shown_variants)) & (aggregate.boundary == "original_fit")]
    scopes = [scope for scope in ["training_held", "examined_validation"] if scope in policy.scope.unique()]
    figure, axes = plt.subplots(len(scopes), 2, figsize=(13, 4 * len(scopes)), layout="constrained", squeeze=False)
    methods = [(basis, mode) for basis in config["basis_methods"] for mode in config["modes"]]
    names = [f"{basis}\n{'paired' if mode == 'bilateral' else 'source'}" for basis, mode in methods]
    colors = {"before": "#99a8b2", "in_sample": "#dfab59", "after": "#176b87"}
    for row_index, scope in enumerate(scopes):
        for column, metric in enumerate(["recall", "negative_retention"]):
            axis = axes[row_index, column]
            width = .8 / len(shown_variants)
            for position, variant in enumerate(shown_variants):
                shift = (position - (len(shown_variants) - 1) / 2) * width
                selected = policy[(policy.scope == scope) & (policy.variant == variant)].set_index(["basis", "mode"])
                values = [selected.loc[key, metric] for key in methods]
                axis.bar(np.arange(len(methods)) + shift, values, width=width * .94, label=variant, color=colors[variant])
            axis.set_xticks(range(len(methods)), names, fontsize=8)
            axis.set_ylim(0, 1)
            axis.set_title(scope.replace("_", " ") + " | " + metric.replace("_", " "))
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
    axes[0, 0].legend()
    figure.suptitle(("Technical smoke | " if directory.name == "smoke" else "") +
        "Frozen task similarity: shared-component intervention\nOriginal fitting threshold retained; procedure means, then seed means")
    figure.savefig(output / "shared_components.png", dpi=150)
    plt.close(figure)
    atomic_write_json(output / "summary.json", dict(status="COMPLETE", procedure_rows=len(table),
        component_rows=len(component_table), results=aggregate.to_dict("records")))
    print(policy.to_csv(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    summarize(args.run, args.directory or args.run)
