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


LABELS = {
    "reference_supcon": "Original / shared threshold",
    "reference_calibrated": "Original / source calibration",
    "sae0445_shared": "0445 SAE / shared threshold",
    "sae0445_calibrated": "0445 SAE / source calibration",
    "sae_fixed_calibrated": "Fixed-action SAE / source calibration",
    "sae_conditional_shared": "Conditional SAE / shared threshold",
    "sae_conditional_calibrated": "Conditional SAE / source calibration",
    "dense_fixed_calibrated": "Fixed-action dense / source calibration",
    "dense_conditional_shared": "Conditional dense / shared threshold",
    "dense_conditional_calibrated": "Conditional dense / source calibration",
    "sae_training_context_calibrated": "Training-context action / source calibration",
    "sae_random0_calibrated": "Norm-matched random 1 / source calibration",
    "sae_random1_calibrated": "Norm-matched random 2 / source calibration",
    "sae_random2_calibrated": "Norm-matched random 3 / source calibration"
}


def render(run, smoke):
    root = run / "smoke" if smoke else run
    path = root / "summary.json"
    summary = read_json(path)
    output = root / "figures"
    output.mkdir(exist_ok=True)
    table = pd.DataFrame(summary["procedure_rows"])
    table.to_csv(output / "procedure_outcomes.csv", index=False)
    methods = [name for name in LABELS if name in set(table.method)]
    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    for row, population in enumerate(("development", "extension")):
        subset = table[table.population == population]
        for column, metric in enumerate(("removal", "retention")):
            ax = axes[row, column]
            for index, method in enumerate(methods):
                values = subset[subset.method == method].groupby("seed")[metric].mean()
                for seed_index, (seed, value) in enumerate(values.items()):
                    ax.scatter(100 * value, index, s=34, color=colors[seed_index],
                               label=f"Seed {seed}" if index == 0 and row == column == 0 else None)
                ax.scatter(100 * values.mean(), index, s=36, marker="|", color="black", zorder=5)
            ax.set_yticks(range(len(methods)), [LABELS[name] for name in methods], fontsize=9)
            ax.invert_yaxis()
            ax.set_title(("Development" if row == 0 else "Previously examined extension") + " | " +
                         ("Repeat-prompt removal" if column == 0 else "Other-prompt retention"), fontsize=12)
            ax.set_xlabel("Percent")
            ax.grid(axis="x", color="#dddddd", linewidth=.7)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(.66, .958))
    fig.suptitle("Confirmation-conditioned component intervention", fontsize=17, y=.995)
    fig.text(.02, .018, "Points show every seed; black marks show their mean. Procedure means define each seed result. "
             "Seeds are repeated fits.\nMethods share the development protection floor and first-prompt constraint. "
             "Extension uses frozen development offsets. Panel x-axis ranges differ.", fontsize=10)
    fig.tight_layout(rect=(0, .075, 1, .95))
    for extension in ("png", "pdf"):
        fig.savefig(output / ("application_outcomes." + extension), dpi=200)
    plt.close(fig)
    rows = []
    for result in summary["held_policy_results"]:
        for video, values in result["per_procedure"].items():
            rows.append(dict(method=result["method"], seed=result["seed"], fold=result["fold"], video=video,
                fixed_delta=values["fixed"] - values["zero"], conditional_delta=values["conditional"] - values["zero"],
                oracle_delta=values["oracle"] - values["zero"]))
    pd.DataFrame(rows).to_csv(output / "held_policy_proxy.csv", index=False)
    manifest = dict(status="COMPLETE", summary_sha256=file_sha256(path), plot_source_sha256=file_sha256(__file__),
        files={p.name: file_sha256(p) for p in output.iterdir() if p.suffix in (".csv", ".png", ".pdf")},
        population="Smoke sources only" if smoke else "Examined application procedures",
        visual_inspection="Pending inspection of the actual rendered image")
    atomic_write_json(output / "manifest.json", manifest)
    atomic_write_json(root / "plot_progress.json", dict(completed=1, total=1, phase="COMPLETE"))
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    render(args.run, args.smoke)
