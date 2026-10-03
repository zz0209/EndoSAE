import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_io import atomic_write_json, read_json
from src.acknowledgement_sae import file_sha256


def summarize(run):
    summary = read_json(run / "summary.json")
    rows = []
    for cell in summary["cells"]:
        identity = {key: cell[key] for key in ("method", "seed", "fold", "arm", "head", "cohort")}
        identity["head_role"] = "evaluation" if cell["head"] == "evaluation" else "training"
        for view, report in cell["views"].items():
            for video, metrics in report["procedures"].items():
                rows.append(dict(**identity, view=view, video=video, **metrics))
    table = pd.DataFrame(rows)
    for metric in ("utility", "recall", "tail"):
        table[metric + "_change"] = table[metric] - table["zero_" + metric]
    table.to_csv(run / "procedure_results.csv", index=False)
    keys = ["method", "seed", "fold", "arm", "head_role", "cohort", "view"]
    per_video = table.groupby(keys + ["video"]).mean(numeric_only=True)
    grouped = per_video.groupby(keys).mean(numeric_only=True).reset_index()
    grouped.to_csv(run / "cell_means.csv", index=False)
    means = grouped.groupby(["method", "arm", "fold", "head_role", "cohort", "view"]).mean(numeric_only=True)
    means.to_csv(run / "seed_means.csv")
    conditions = [("training", "fit"), ("training", "held"), ("evaluation", "held")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, view in zip(axes, ("region", "canonical")):
        selected = grouped[(grouped.view == view) & (grouped.fold != "full_training")]
        for (method, arm), subset in selected.groupby(["method", "arm"]):
            values = [subset[(subset.head_role == head) & (subset.cohort == cohort)].utility_change.mean() for head, cohort in conditions]
            label = method.replace("_edit", "") + "/" + arm.replace("_decoder", "")
            axis.plot(range(3), values, marker="o", label=label)
        axis.axhline(0, color="grey", linewidth=.8)
        axis.set_xticks(range(3), ["Fitted procedures\ntraining heads", "Held procedures\ntraining heads", "Held procedures\nevaluation head"])
        axis.set(title=view + " source views", ylabel="Utility change relative to zero edit")
        axis.legend(fontsize=7)
    fig.text(.01, .01, "Same saved interventions. All three outer folds and seeds; within-cell procedures receive equal weight. Fitted-procedure results are in-sample.", fontsize=8)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(run / "head_transfer.png", dpi=180)
    plt.close(fig)
    atomic_write_json(run / "analysis_complete.json", dict(status="COMPLETE", cells=len(summary["cells"]),
        procedure_rows=len(table), source_sha256=file_sha256(__file__), summary_sha256=file_sha256(run / "summary.json")))
    print(means[["utility_change", "recall_change", "tail_change"]].to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    summarize(parser.parse_args().run)
