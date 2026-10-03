import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_io import atomic_write_json, read_json
from src.acknowledgement_sae import file_sha256


def prepare(run, phase, evaluation_seconds):
    if phase == "config":
        if run.exists():
            raise FileExistsError(run)
        run.mkdir(parents=True)
        config = read_json(ROOT / "results/runs/20261003T0945Z_template_component_adaptation_v1/config.json")
        config.update(run_id=run.name, run_dir=str(run), method_family="observed_view_component_margin",
            predecessor_run="results/runs/20261003T0945Z_template_component_adaptation_v1",
            positive_quantile=.1, view_cache=str(run / "observed_views"))
        atomic_write_json(run / "config.json", config)
        atomic_write_json(run / "protocol.json", dict(
            question="Does preserving observed source variation improve the transfer of sparse source adaptation?",
            evidence="0945 improved high-protection recall mainly in one training procedure, while full-video SAE removal declined in4/5 development and5/5 examined-extension procedures. Its available objective was positive at every application source. A single-source self-similarity penalty does not measure preservation of other views.",
            hypothesis="Using the lower tail of similarity to already observed source frames in the adaptation objective preserves useful identity variation while reducing training-cohort confusion.",
            mathematics="For normalized memory m and source u, 1-m dot u equals one half squared Euclidean displacement and is second-order for small tangent moves. Similarity to another view changes at first order. The observed-view objective is Q10(m dot P)-Q99_weighted(m dot N), where P contains all nonempty frame-wise ROI pools within the existing source clip and N is the unchanged training-only cohort. With P={u}, objective improvement equals the0945 B-H up to numerical normalization error.",
            mechanism="Measure exact unit suppression of each active component, rank positive margin improvements, form the unchanged zero plus4/16/64 components at0.25/0.5/1 strengths, and recompute the joint objective. Select the best including zero. No gain from future labels or threshold prediction enters the choice.",
            data="Existing19 training procedures and three inherited excluded-procedure heads/dictionaries/cohorts. Training frame views use only each existing source clip and its current ROI mask. Actual application views use the unchanged eight causal frames ending at acknowledgement, with original base-track support. Nonempty frames are retained; no ground-truth boxes or later frames enter application adaptation. Existing10 application procedures remain previously examined; reserved data untouched.",
            comparisons="Original, global0445 SAE,0945 self-only SAE, observed-view SAE, observed-view dense, equal-frame embedding mean, same-view LinearSVC, and three gain-multiset/norm-matched random SAE controls. Held diagnosis compares original, self-only, view-based, mean and SVM within every dictionary/seed/fold. Ordinary alternatives receive the same source views and training cohort.",
            svm="C10 squared-hinge L2 primal solver. Uniform positive-view weights and inherited equal-procedure/lesion negative weights; total class masses equal and total sample-weight scale fixed to bank_size+1, preserving single-source regularization strength.",
            selection="All definitions fixed before results. Source criterion Q10/Q99 is fixed; candidate budget inherited. Global development thresholds satisfy the original other-prompt retention floor and first-prompt requirement. Examined extension uses frozen thresholds.",
            decision="Improved future-identity utility and application removal at required protection support the observed-variation mechanism. Dense/mean/SVM comparisons locate any value specific to sparse interventions. Similar decline despite actual-view preservation rejects this concrete proxy; do not tune its quantiles on application outcomes. Inspect paired effects to decide whether source support, cohort relevance or representation is the next meaningful object.",
            budget="One CPU thread, no GPU; source-view pooling for19 training procedures,18 held jobs, three10-condition full-video evaluations, summary and figures. Normal E then D leases for pooling and application evaluation. Source/view/job checkpoints, frozen hashes and real resume test.",
            references=["SRC-TPT-2022", "SRC-TEMPLATE-ADAPTATION-FG2017", "SRC-SAE-FT-2026", "SRC-EXPOSE-2026", "SRC-POLYP-REID-2023", "SRC-ENDOFINDER-2025"]))
        print(run, flush=True)
        return
    config = read_json(run / "config.json")
    python = str(ROOT / "artifacts/environments/modern/Scripts/python.exe")
    target = ("\\\\?\\" if os.name == "nt" else "") + str(run.resolve())
    command = [python, "scripts/evaluate_observed_view_adaptation.py", "--run", target]
    stages = []
    for name, label, units, seconds, resources in (("views", "训练来源的逐帧表示", 19, 90, ["disk-e-io", "disk-d-io"]),
            ("held", "观察视图与未来身份检验", 18, 180, [])):
        stages.append(dict(id=name, label=label, kind=name, command=command + ["--phase", name, "--resume"],
            progress=str(run / (name + "_progress.json")), output=str(run / (name + "_summary.json")),
            units=units, unit="组完成结果", estimate_seconds=seconds, resources=resources))
    for seed in config["seeds"]:
        stages.append(dict(id=f"evaluate_{seed}", label=f"种子 {seed} · 观察视图完整评价", kind="evaluate",
            command=command + ["--phase", "evaluate", "--seed", str(seed), "--resume"],
            progress=str(run / "evaluation" / f"seed{seed}" / "progress.json"),
            output=str(run / "evaluation" / f"seed{seed}" / "summary.json"), units=10, unit="个procedure",
            estimate_seconds=evaluation_seconds, resources=["disk-e-io", "disk-d-io"]))
    stages.append(dict(id="summary", label="完整比较与逐procedure图表", kind="summary",
        command=command + ["--phase", "summary", "--resume"], progress=str(run / "summary_progress.json"),
        output=str(run / "figures/manifest.json"), units=1, unit="组分析", estimate_seconds=25, resources=[]))
    path = run / "batch_plan.json"
    if path.exists():
        raise FileExistsError(path)
    files = ["src/observed_view_adaptation.py", "scripts/evaluate_observed_view_adaptation.py",
        "scripts/prepare_observed_view_batch.py", "scripts/run_sae_acknowledgement_batch.py"]
    atomic_write_json(path, dict(question=read_json(run / "protocol.json")["question"], stages=stages,
        config_sha256=file_sha256(run / "config.json"), protocol_sha256=file_sha256(run / "protocol.json"),
        source_hashes={name: file_sha256(ROOT / name) for name in files},
        estimate_basis="Real smoke throughput and preceding three full-video evaluations; monitor updates with observed progress."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("config", "plan"), required=True)
    parser.add_argument("--evaluation-seconds", type=float, default=480.)
    args = parser.parse_args()
    prepare(args.run, args.phase, args.evaluation_seconds)
