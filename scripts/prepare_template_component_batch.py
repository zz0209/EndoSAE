import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_io import atomic_write_json, read_json
from src.acknowledgement_sae import file_sha256


def prepare(run, phase, evaluation_seconds, held_seconds):
    if phase == "config":
        if run.exists():
            raise FileExistsError(run)
        run.mkdir(parents=True)
        config = read_json(ROOT / "results/runs/20261003T0830Z_conditioned_component_policy_v1/config.json")
        config.update(run_id=run.name, run_dir=str(run), method_family="source_template_component_adaptation",
            predecessor_run="results/runs/20261003T0830Z_conditioned_component_policy_v1")
        for key in ("ridge_alpha", "discovery_run"):
            del config[key]
        atomic_write_json(run / "config.json", config)
        atomic_write_json(run / "protocol.json", dict(
            question="Can direct source-template adaptation in the deployed identity head produce useful sparse component interventions?",
            motivation="0830 learned intervention advantages did not transfer across held procedures; source-threshold R2 was negative in every fold, and full-video source calibration reduced removal and extension protection. Source information may still support direct local adaptation.",
            hypothesis="Components whose removal reduces high similarity to other training identities while preserving the observed source identity can suppress shared appearance and improve repeat-prompt removal at the original protection requirement.",
            available_information="Actual acknowledged source raw descriptor and token codes, frozen identity head, fixed training-only canonical lesion cohort. Future application identities, frames and scores never enter source adaptation. No extra source view or label is supplied to any method.",
            mechanism="For each active component, perform an exact unit suppression in raw space and encode with the deployed head. B is the reduction in procedure/lesion-weighted cohort 99th-percentile cosine; H is one minus cosine to the unedited source memory. Rank positive B-H with positive B. Construct the inherited ten k/strength actions and recompute their combined B-H, including exact zero. Select its maximum, with zero first in ties. Components, action and effects are saved for each source.",
            weighting="Cohort: equal procedure, equal lesion within procedure, equal descriptor within lesion. Held proxy: equal source views, equal sources within procedure, equal eligible procedure.",
            baseline="Original identity head and fixed0445 sparse intervention; ordinary dense dictionary uses the identical local adaptation rule and action budget. One-way template LinearSVC uses the same original source and negative cohort, C10, squared hinge, L2, bias, equal class mass and the same within-negative cohort weights. Sigmoid of functional margin bounds scores without estimating probabilities. Three sparse random controls preserve gain multiset and raw edit norm.",
            svm_provenance="Crosswhite et al., FG2017, template adaptation. One-way source adaptation fits the online task; symmetric probe adaptation would alter the query consumer and is not used. This is an adapted established baseline, not an official reproduction.",
            selection="Local action selection uses only available B-H; no outcome regression or per-source threshold prediction. Main model and action rule are fixed before this run. Global prompt thresholds use the original development retention floor and all first prompts; extension reuses them unchanged.",
            data="Existing19 training procedures; three existing held folds exclude their procedures from dictionary, identity head and cohort. Existing10 examined application procedures. Both application populations retain their exposed status. Reserved independent data are not read.",
            scope="Two dictionaries, three seeds;18 held-procedure jobs; three8-condition full-video evaluations; procedure tables and figures. Held results diagnose the predefined local rule against original and SVM; they are not used to tune it. The full-video comparison includes the historical global action.",
            resources="One CPU thread, no GPU. Reuse saved pooled training arrays and actual source tokens; normal E then D I/O leases for complete application evaluation. Checkpoint each held source/view and each adapted application source, reuse existing query-evaluation checkpoints.",
            tests="Real smoke with both dictionaries and development/extension sources; independent direct single-component and grouped residual-edit check; manual SVM score check against exported coefficients; actual source-boundary exit75 and resume compared numerically with continuous execution.",
            decision="Benefit over original/global and dense plus targeted-versus-random advantage supports source-local sparse intervention. SVM-only benefit identifies generic template adaptation value. Failure of both cohort methods questions cohort relevance; retain the specific negative result and use saved individual effects to decide the next evidence-driven experiment. Independent confirmation is required for a final application claim.",
            references=["SRC-TEMPLATE-ADAPTATION-FG2017", "SRC-ARAD-OUTPUT-STEERING-2025", "SRC-ENDOFINDER-2025", "SRC-SKLEARN-LINEARSVC-191"] ))
        print(run, flush=True)
        return
    config = read_json(run / "config.json")
    python = str(ROOT / "artifacts/environments/modern/Scripts/python.exe")
    target = ("\\\\?\\" if os.name == "nt" else "") + str(run.resolve())
    command = [python, "scripts/evaluate_template_component_adaptation.py", "--run", target]
    stages = [dict(id="held", label="可用源信息的成分作用检验", kind="held", command=command + ["--phase", "held", "--resume"],
        progress=str(run / "held_progress.json"), output=str(run / "held_summary.json"), units=18, unit="组留出procedure比较",
        estimate_seconds=held_seconds, resources=[])]
    for seed in config["seeds"]:
        stages.append(dict(id=f"evaluate_{seed}", label=f"种子 {seed} · 模板适配完整视频评价", kind="evaluate",
            command=command + ["--phase", "evaluate", "--seed", str(seed), "--resume"],
            progress=str(run / "evaluation" / f"seed{seed}" / "progress.json"),
            output=str(run / "evaluation" / f"seed{seed}" / "summary.json"), units=10, unit="个procedure",
            estimate_seconds=evaluation_seconds, resources=["disk-e-io", "disk-d-io"]))
    stages.extend([dict(id="summary", label="应用与逐procedure结果", kind="summary", command=command + ["--phase", "summary", "--resume"],
        progress=str(run / "summary_progress.json"), output=str(run / "summary.json"), units=1, unit="份结果", estimate_seconds=20, resources=[]),
        dict(id="plot", label="应用及可测成分作用图表", kind="plot", command=[python, "scripts/plot_template_component_adaptation.py", "--run", target],
        progress=str(run / "plot_progress.json"), output=str(run / "figures/manifest.json"), units=1, unit="组图表", estimate_seconds=15, resources=[])])
    path = run / "batch_plan.json"
    if path.exists():
        raise FileExistsError(path)
    names = ["src/template_component_adaptation.py", "scripts/evaluate_template_component_adaptation.py",
             "scripts/plot_template_component_adaptation.py", "scripts/prepare_template_component_batch.py", "scripts/run_sae_acknowledgement_batch.py"]
    atomic_write_json(path, dict(question=read_json(run / "protocol.json")["question"], stages=stages,
        config_sha256=file_sha256(run / "config.json"), protocol_sha256=file_sha256(run / "protocol.json"),
        source_hashes={name: file_sha256(ROOT / name) for name in names}, estimate_basis="Measured real smoke and previous complete-cohort throughput; updated by live progress."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("config", "plan"), required=True)
    parser.add_argument("--evaluation-seconds", type=float, default=600.)
    parser.add_argument("--held-seconds", type=float, default=300.)
    args = parser.parse_args()
    prepare(args.run, args.phase, args.evaluation_seconds, args.held_seconds)
