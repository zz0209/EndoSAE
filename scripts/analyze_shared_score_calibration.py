import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_io import atomic_write_json, read_json
from src.conditional_component_policy import source_weights
from src.acknowledgement_sae import file_sha256


def analyze(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for run in runs:
        summary = read_json(run / "training_summary.json")
        for job in summary["outputs"]:
            folder = Path(job["folder"])
            report = read_json(folder / "held.json")
            weights = source_weights(report["sources"])
            with np.load(folder / "held.npz", allow_pickle=False) as data:
                for index, condition in enumerate(("original", "edited")):
                    positives, negatives, wp, wn = [], [], [], []
                    for i, (row, weight) in enumerate(zip(report["sources"], weights)):
                        split = len(row["positives"])
                        scores = data[f"scores_{i}"][index]
                        positives.extend(scores[:split])
                        negatives.extend(scores[split:])
                        wp.extend([weight / split] * split)
                        wn.extend([weight / (len(scores) - split)] * (len(scores) - split))
                    positives, negatives = np.array(positives), np.array(negatives)
                    wp, wn = np.array(wp), np.array(wn)
                    threshold = float(np.quantile(negatives, .99, weights=wn, method="inverted_cdf"))
                    local_name = "zero_threshold" if index == 0 else "threshold"
                    local = np.array([r[local_name] for r in report["sources"]])
                    rows.append(dict(run=run.name, **{k: job[k] for k in ("method", "seed", "fold", "arm")},
                        condition=condition, local_recall=report["means"]["zero_recall" if index == 0 else "recall"],
                        shared_recall=float(wp @ (positives > threshold)),
                        shared_utility=float(wp @ (1 / (1 + np.exp(-(positives - threshold) / .05)))),
                        shared_false_match=float(wn @ (negatives > threshold)), shared_threshold=threshold,
                        local_threshold_min=float(local.min()), local_threshold_max=float(local.max()),
                        local_threshold_sd=float(np.sqrt(weights @ (local - weights @ local) ** 2)),
                        procedures=len(report["procedure_results"]), source_views=len(weights),
                        excluded_sources=len(report["excluded"])))
        print("CALIBRATION_SCORED", run.name, len(summary["outputs"]), flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "jobs.csv", index=False)
    keys = ["run", "method", "seed", "fold", "arm"]
    paired = table[table.condition == "edited"].merge(table[table.condition == "original"], on=keys,
        suffixes=("", "_original"), validate="one_to_one")
    for metric in ("local_recall", "shared_recall", "shared_utility", "local_threshold_sd"):
        paired[metric + "_change"] = paired[metric] - paired[metric + "_original"]
    paired.to_csv(output / "paired_jobs.csv", index=False)
    means = paired.groupby(["run", "method", "fold", "arm"])[[c for c in paired if c.endswith("_change")]].mean()
    means.to_csv(output / "seed_means.csv")
    fig, axes = plt.subplots(1, len(runs), figsize=(6 * len(runs), 5), squeeze=False)
    for ax, run in zip(axes[0], runs):
        for (method, arm), data in paired[paired.run == run.name].groupby(["method", "arm"]):
            ax.scatter(data.local_recall_change * 100, data.shared_recall_change * 100,
                label=method.replace("_edit", "") + "/" + arm.replace("_decoder", ""), alpha=.8)
        ax.axhline(0, color="gray", lw=.7)
        ax.axvline(0, color="gray", lw=.7)
        ax.set_title(run.name.split("_")[1].capitalize() + " training")
        ax.set_xlabel("Source-local recall change (percentage points)")
        ax.set_ylabel("Shared-threshold recall change (percentage points)")
        ax.legend(fontsize=8)
    fig.text(.03, .015, "Saved held scores; all seeds and scopes. Each point is one fitted model. Both metrics use held-label thresholds.\n"
        "Equal procedures, sources and views; these are diagnostic ceilings, not deployment results.", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, 1))
    fig.savefig(output / "shared_calibration.png", dpi=180)
    plt.close(fig)
    atomic_write_json(output / "manifest.json", dict(status="COMPLETE", rows=len(rows),
        inputs={str(run / "training_summary.json"): file_sha256(run / "training_summary.json") for run in runs},
        source_sha256=file_sha256(__file__),
        scope="Previously examined held data; posthoc diagnostic only. No training or model selection."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.runs, args.output)
