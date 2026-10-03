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


def analyze(run):
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    if (output / "complete.json").exists():
        raise FileExistsError(output)
    training = read_json(run / "training_summary.json")
    summary = read_json(run / "summary.json")
    rows, procedures = [], []
    for job in training["outputs"]:
        history = read_json(Path(job["folder"]) / "history.json")
        identity = {key: job[key] for key in ("method", "seed", "fold", "arm")}
        row = dict(**identity, **job["held"])
        for term in ("canonical_loss", "tail_loss", "anchor_loss"):
            row[term + "_initial"] = history[0][term]
            row[term + "_end"] = history[-1][term]
        rows.append(row)
        procedures.extend(dict(**identity, video=video, **metrics)
                          for video, metrics in job["procedure_results"].items())
    jobs, held = pd.DataFrame(rows), pd.DataFrame(procedures)
    for table in (jobs, held):
        for metric in ("utility", "recall"):
            table[metric + "_change"] = table[metric] - table["zero_" + metric]
    jobs.to_csv(output / "jobs.csv", index=False)
    held.to_csv(output / "held_procedures.csv", index=False)
    application = pd.DataFrame(summary["procedure_rows"])
    original = application[application.method == "reference_supcon"].drop(columns="method")
    changes = application.merge(original, on=["population", "seed", "video"],
                                suffixes=("", "_original"), validate="many_to_one")
    for metric in ("removal", "retention", "first_prompt"):
        changes[metric + "_change"] = changes[metric] - changes[metric + "_original"]
    changes.to_csv(output / "application_procedures.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    for (method, arm), group in jobs[jobs.fold != "full_training"].groupby(["method", "arm"]):
        label = method.replace("_edit", "") + "/" + arm.replace("_decoder", "")
        axes[0].scatter(group.tail_loss_initial - group.tail_loss_end, group.utility_change, label=label)
    axes[0].axhline(0, color="grey", linewidth=.8)
    axes[0].set(xlabel="Training tail-loss decrease", ylabel="Held utility change", title="All 36 outer-held fits")
    axes[0].legend(fontsize=7)
    names = ["sae_fixed", "sae_refined", "dense_fixed", "dense_refined"]
    for axis, population in zip(axes[1:], ("development", "extension")):
        grouped = changes[(changes.population == population) & changes.method.isin(names)].groupby(["method", "video"]).mean(numeric_only=True)
        for index, name in enumerate(names):
            axis.scatter(np.full(len(grouped.loc[name]), index), 100 * grouped.loc[name].removal_change, s=25)
        axis.axhline(0, color="grey", linewidth=.8)
        axis.set_xticks(range(4), [n.replace("_", "\n") for n in names])
        axis.set(title=population + ": five procedures", ylabel="Removal change (percentage points)")
    fig.text(.015, .01, "Previously examined populations. Application points average all three seeds within each procedure; original calibration is unchanged.", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output / "transfer.png", dpi=180)
    plt.close(fig)
    atomic_write_json(output / "complete.json", dict(status="COMPLETE", jobs=len(jobs),
        held_procedure_rows=len(held), application_rows=len(changes), source_sha256=file_sha256(__file__),
        inputs={name: file_sha256(run / name) for name in ("training_summary.json", "summary.json")}))
    print(jobs.groupby(["method", "arm", "fold"]).mean(numeric_only=True).to_string())
    print(changes.groupby(["population", "method", "video"])[["removal_change", "retention_change", "first_prompt_change"]].mean().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    analyze(parser.parse_args().run)
