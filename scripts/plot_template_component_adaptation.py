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


def render(run, smoke):
    root = run / "smoke" if smoke else run
    summary = read_json(root / "summary.json")
    output = root / "figures"
    output.mkdir(exist_ok=True)
    table = pd.DataFrame(summary["procedure_rows"])
    table.to_csv(output / "procedure_outcomes.csv", index=False)
    labels = dict(reference_supcon="Original", sae0445="Global SAE intervention", sae_adapted="Source-adapted SAE",
        dense_adapted="Source-adapted dense", template_svm="Template SVM", sae_random0="Norm-matched random 1",
        sae_random1="Norm-matched random 2", sae_random2="Norm-matched random 3")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for i, population in enumerate(("development", "extension")):
        for j, metric in enumerate(("removal", "retention")):
            ax = axes[i, j]
            subset = table[table.population == population]
            for number, (method, label) in enumerate(labels.items()):
                values = subset[subset.method == method].groupby("seed")[metric].mean()
                for seed_index, (seed, value) in enumerate(values.items()):
                    ax.scatter(100 * value, number, color=["#0072B2", "#D55E00", "#009E73"][seed_index],
                        label=str(seed) if number == i == j == 0 else None)
                ax.scatter(100 * values.mean(), number, marker="|", color="black")
            ax.set_yticks(range(len(labels)), list(labels.values()), fontsize=9)
            ax.set_xlabel("Percent")
            ax.set_title(("Development" if i == 0 else "Examined extension") + " | " +
                ("Repeat removal" if j == 0 else "Other-prompt retention"))
            ax.grid(axis="x", alpha=.25)
            ax.spines[["right", "top"]].set_visible(False)
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=len(names), frameon=False, bbox_to_anchor=(.65, .96))
    fig.suptitle("Source-template component adaptation" + (" — real smoke only" if smoke else ""), fontsize=16)
    fig.text(.02, .025, "Points show all seeds; black marks show means. Equal procedure weighting. Panel scales differ.\n"
        "Common development protection requirement; extension uses frozen thresholds. Seeds are repeated fits, not independent procedures.", fontsize=9)
    fig.tight_layout(rect=(0, .08, 1, .93))
    for extension in ("png", "pdf"):
        fig.savefig(output / ("application_outcomes." + extension), dpi=180)
    plt.close(fig)
    rows = []
    for seed in summary["seeds"]:
        diagnostics = read_json(root / "evaluation" / f"seed{seed}" / "source_diagnostics.json")
        for source in diagnostics["sources"]:
            if source["status"] == "COMPLETE":
                for method in ("sparse_edit", "dense_edit"):
                    rows.append(dict(seed=seed, method=method, population=source["population"], video=source["video"],
                        episode=source["episode"], **{key: source["decisions"][method][key] for key in
                        ("choice", "eligible_components", "active_components", "gain", "component_count")}))
    pd.DataFrame(rows).to_csv(output / "source_component_effects.csv", index=False)
    atomic_write_json(output / "manifest.json", dict(status="COMPLETE", summary_sha256=file_sha256(root / "summary.json"),
        source_sha256=file_sha256(__file__), files={p.name: file_sha256(p) for p in output.iterdir() if p.suffix in (".png", ".pdf", ".csv")}))
    atomic_write_json(root / "plot_progress.json", dict(completed=1, total=1, phase="COMPLETE"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    render(args.run, args.smoke)
