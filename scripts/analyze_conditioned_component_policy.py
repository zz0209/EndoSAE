import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
from collections import Counter
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
from src.conditional_component_policy import FEATURE_NAMES, source_weights


def analyze(run):
    output = run / "analysis"
    output.mkdir(exist_ok=True)
    if (output / "diagnostics.json").exists():
        raise FileExistsError("Completed analysis is immutable")
    config = read_json(run / "config.json")
    result = read_json(run / "summary.json")
    if result["status"] != "COMPLETE":
        raise ValueError("Complete application results are required")
    tables, feature_rows, application = [], [], []
    inputs = {str(run / "summary.json"): file_sha256(run / "summary.json")}
    for method in config["methods"]:
        for seed in config["seeds"]:
            scopes = [(f"fold{fold}", run / "inner_folds" / method / f"seed{seed}" / f"fold{fold}")
                      for fold in range(3)] + [("full", run / "fit" / method / f"seed{seed}")]
            for scope, folder in scopes:
                for split in (["fitting", "held"] if scope != "full" else ["fitting"]):
                    meta = read_json(folder / (split + ".json"))["metadata"]
                    path = folder / (split + ".npz")
                    inputs[str(path)] = file_sha256(path)
                    with np.load(path, allow_pickle=False) as saved:
                        data = {name: saved[name] for name in ("features", "thresholds", "utility", "chosen",
                                                               "predicted_advantage", "predicted_threshold")}
                    weights = source_weights(meta)
                    choice, utility = data["chosen"], data["utility"]
                    index = np.arange(len(meta))
                    fixed = read_json(folder / "policy.json")["fixed_action"]
                    actual = data["thresholds"][:, 0]
                    predicted = data["predicted_threshold"][:, 0]
                    mse = weights @ ((actual - predicted) ** 2)
                    variance = weights @ ((actual - weights @ actual) ** 2)
                    tables.append(dict(method=method, seed=seed, scope=scope, split=split,
                        procedures=len({r["video_id"] for r in meta}), rows=len(meta),
                        zero=float(weights @ utility[:, 0]), fixed=float(weights @ utility[:, fixed]),
                        conditional=float(weights @ utility[index, choice]), oracle=float(weights @ utility.max(1)),
                        predicted_gain=float(weights @ data["predicted_advantage"][index, choice]),
                        realized_gain=float(weights @ (utility[index, choice] - utility[:, 0])),
                        threshold_mean=float(weights @ actual), predicted_threshold_mean=float(weights @ predicted),
                        threshold_mae=float(weights @ np.abs(actual - predicted)), threshold_r2=float(1 - mse / variance),
                        actions=dict(Counter(choice.tolist()))))
                    for column, name in enumerate(FEATURE_NAMES):
                        values = data["features"][:, 0, column]
                        feature_rows.append(dict(method=method, seed=seed, scope=scope, split=split, feature=name,
                            mean=float(weights @ values), std=float(np.sqrt(weights @ ((values - weights @ values) ** 2))),
                            minimum=float(values.min()), maximum=float(values.max())))
            policies = read_json(run / "evaluation" / f"seed{seed}" / "source_policies.json")
            for row in policies["sources"]:
                if row["status"] != "COMPLETE":
                    continue
                decision = row["decisions"][method]
                path = run / "evaluation" / f"seed{seed}" / "source_policies" / row["population"] / row["video"] / row["episode_id"] / "predictions.npz"
                with np.load(path, allow_pickle=False) as saved:
                    features = saved[method + "__features"]
                chosen = decision["conditional_action"]
                application.append(dict(method=method, seed=seed, population=row["population"], video=row["video"],
                    episode=row["episode_id"], action=chosen, threshold=decision["predicted_threshold"][0],
                    predicted_gain=decision["predicted_advantage"][chosen],
                    **{name: float(features[0, i]) for i, name in enumerate(FEATURE_NAMES)}))
    table, feature_table, application_table = pd.DataFrame(tables), pd.DataFrame(feature_rows), pd.DataFrame(application)
    table.to_csv(output / "policy_generalization.csv", index=False)
    feature_table.to_csv(output / "feature_ranges.csv", index=False)
    application_table.to_csv(output / "application_context.csv", index=False)
    procedures = pd.DataFrame(result["procedure_rows"])
    contrasts = []
    for population in ("development", "extension"):
        for method in sorted(procedures.method.unique()):
            selected = procedures[(procedures.population == population) & (procedures.method == method)]
            baseline = procedures[(procedures.population == population) & (procedures.method == "reference_supcon")]
            merged = selected.merge(baseline, on=["seed", "video"], suffixes=("", "_base"), validate="one_to_one")
            for row in merged.to_dict("records"):
                contrasts.append(dict(population=population, method=method, seed=row["seed"], video=row["video"],
                    removal_delta=row["removal"] - row["removal_base"], retention_delta=row["retention"] - row["retention_base"],
                    first_delta=row["first_prompt"] - row["first_prompt_base"]))
    pd.DataFrame(contrasts).to_csv(output / "procedure_contrasts.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for mi, method in enumerate(config["methods"]):
        held = table[(table.method == method) & (table.split == "held")]
        for seed in config["seeds"]:
            subset = held[held.seed == seed]
            axes[0].scatter(subset.predicted_gain, subset.realized_gain, label=f"{method} / {seed}", marker=["o", "s"][mi])
        for scope in ("fitting", "held"):
            subset = table[(table.method == method) & (table.split == scope) & (table.scope != "full")]
            x = mi * 2 + int(scope == "held")
            axes[1].scatter(np.full(len(subset), x), subset.threshold_r2, alpha=.8)
    axes[0].axhline(0, color="grey", linewidth=.8)
    axes[0].axvline(0, color="grey", linewidth=.8)
    axes[0].set(xlabel="Predicted utility gain", ylabel="Observed held-procedure utility gain", title="Conditional action transfer")
    axes[0].legend(fontsize=7)
    axes[1].axhline(0, color="grey", linewidth=.8)
    axes[1].set_xticks(range(4), ["SAE fit", "SAE held", "Dense fit", "Dense held"])
    axes[1].set(ylabel="Procedure-weighted threshold R²", title="Source calibration transfer")
    fig.text(.02, .015, "Each point is one seed/fold model. Folds are disjoint procedures; seeds repeat the same folds. Utility is a training-label proxy.", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(output / "policy_transfer.png", dpi=180)
    fig.savefig(output / "policy_transfer.pdf")
    plt.close(fig)
    compact = dict(inputs=inputs, source_sha256=file_sha256(__file__), fit_and_held=tables,
        application_actions={f"{m}/{s}/{p}": dict(Counter(application_table[(application_table.method == m) &
            (application_table.seed == s) & (application_table.population == p)].action.tolist()))
            for m in config["methods"] for s in config["seeds"] for p in ("development", "extension")})
    atomic_write_json(output / "diagnostics.json", compact)
    print(table.groupby(["method", "split"])[["predicted_gain", "realized_gain", "threshold_mae", "threshold_r2"]].mean().to_string())
    print("COMPLETE", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    analyze(parser.parse_args().run)
