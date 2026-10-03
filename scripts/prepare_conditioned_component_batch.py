import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_io import atomic_write_json, read_json
from src.acknowledgement_sae import file_sha256


def prepare(run, evaluation_seconds):
    config = read_json(run / "config.json")
    python = str(ROOT / "artifacts/environments/modern/Scripts/python.exe")
    evaluation_run = ("\\\\?\\" if os.name == "nt" else "") + str(run.resolve())
    stages = [dict(id="fit", label="确认条件与干预作用学习", kind="fit",
        command=[python, "scripts/train_conditioned_component_policy.py", "--config", str(run / "config.json"), "--resume"],
        progress=str(run / "training_progress.json"), output=str(run / "training_summary_all.json"),
        units=24, unit="组模型与独立procedure比较", estimate_seconds=30, resources=[])]
    for seed in config["seeds"]:
        stages.append(dict(id=f"evaluate_{seed}", label=f"种子 {seed} · 完整视频提示评价", kind="evaluate",
            command=[python, "scripts/evaluate_conditioned_component_policy.py", "--run", evaluation_run,
                     "--phase", "evaluate", "--seed", str(seed), "--resume"],
            progress=str(run / "evaluation" / f"seed{seed}" / "progress.json"),
            output=str(run / "evaluation" / f"seed{seed}" / "summary.json"),
            units=10, unit="个procedure", estimate_seconds=evaluation_seconds, resources=["disk-e-io", "disk-d-io"]))
    stages.append(dict(id="summary", label="全部方法与逐procedure结果", kind="summary",
        command=[python, "scripts/evaluate_conditioned_component_policy.py", "--run", evaluation_run, "--phase", "summary", "--resume"],
        progress=str(run / "summary_progress.json"), output=str(run / "summary.json"),
        units=1, unit="份结果", estimate_seconds=30, resources=[]))
    stages.append(dict(id="plot", label="应用结果与成分作用图表", kind="plot",
        command=[python, "scripts/plot_conditioned_component_policy.py", "--run", evaluation_run],
        progress=str(run / "plot_progress.json"), output=str(run / "figures/manifest.json"),
        units=1, unit="组图表", estimate_seconds=20, resources=[]))
    path = run / "batch_plan.json"
    if path.exists():
        raise FileExistsError(path)
    files = ["src/conditional_component_policy.py", "scripts/train_conditioned_component_policy.py",
             "scripts/evaluate_conditioned_component_policy.py", "scripts/plot_conditioned_component_policy.py",
             "scripts/run_sae_acknowledgement_batch.py", "scripts/prepare_conditioned_component_batch.py"]
    atomic_write_json(path, dict(question=read_json(run / "protocol.json")["question"], stages=stages,
        config_sha256=file_sha256(run / "config.json"), protocol_sha256=file_sha256(run / "protocol.json"),
        source_hashes={name: file_sha256(ROOT / name) for name in files},
        estimate_basis="Four complete smoke policy jobs took2.36seconds. Two real application sources took27.97seconds; full evaluation includes28sources with13development complete curves. Provision for source heterogeneity and aggregation; monitor updates from actual progress."))
    print(path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation-seconds", type=float, required=True)
    args = parser.parse_args()
    prepare(args.run, args.evaluation_seconds)
