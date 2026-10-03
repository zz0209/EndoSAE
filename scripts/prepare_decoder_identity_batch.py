import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_io import atomic_write_json, read_json
from src.acknowledgement_sae import file_sha256


def prepare(run, phase, training_seconds):
    if phase == "config":
        if run.exists():
            raise FileExistsError(run)
        run.mkdir(parents=True)
        config = read_json(ROOT / "results/runs/20261003T1030Z_observed_view_component_v1/config.json")
        discovery = read_json(ROOT / "results/runs/20260930T0630Z_crossfit_component_discovery_v1/config.json")
        config.update({key: discovery[key] for key in ("discovery_head_folds", "head_steps", "head_seed", "head_config")})
        config.update(run_id=run.name, run_dir=str(run), method_family="decoder_identity_tail",
            discovery_heads_root=str(Path(discovery["run_dir"]) / "discovery_heads"),
            arms=["fixed_decoder", "trainable_decoder"], steps=500, smoke_steps=8,
            gain_bound=.5, gain_learning_rate=.01, decoder_learning_rate=.00005,
            identity_weight=.1, decoder_anchor_weight=1., gradient_clip_norm=5., checkpoint_every=25,
            device="cpu", threads=1)
        atomic_write_json(run / "config.json", config)
        atomic_write_json(run / "protocol.json", dict(
            question="Do task-trained decoder directions improve high-protection source-memory edits beyond fixed dictionary gains?",
            motivation="The complete observed-view source proxy failed. Saved fixed-candidate hindsight utilities show a smaller sparse action opportunity than the ordinary dictionary. This motivates an intervention-direction comparison, without interpreting the hindsight result as a full capacity bound.",
            hypothesis="With encoding and downstream identity heads fixed, direction refinement can improve future identity separation beyond gains along the original reconstruction directions.",
            formula="x_edit=x+scale*((mean_token_code * .5*tanh(theta)) @ D.T). Loss=mean standardized canonical-source squared error + .1*mean softplus((q99(other-identity scores)-same-identity scores)/.05) + mean squared pooled reconstruction drift from the initial decoder.",
            training="Full-batch fitting over existing chronological identity labels and all seven valid source region views. Equal views within source, sources within procedure, then procedures. Each fitting partition uses the existing identity head trained without that partition; all training heads and dictionary fitting exclude outer-held procedures. Encoder and all biases remain exact; decoder columns retain unit norm. Adam gain lr .01, decoder lr .00005,500 steps, checkpoint every25. No checkpoint selection.",
            comparisons=["original", "historical global SAE", "SAE fixed decoder", "SAE trained decoder", "ordinary fixed decoder", "ordinary trained decoder", "three gain-permuted norm-matched SAE trained-decoder controls"],
            budget="48 fitting jobs: two dictionaries, two decoder arms, three seeds, three outer folds plus full training; three complete application evaluations. CPU single thread, no GPU. Existing E then D leases for full application I/O. No independent reserved data access.",
            endpoints="Primary: complete-procedure repeat-prompt removal at the unchanged common development protection requirement; extension inherits thresholds. Secondary: other and first-prompt protection, all held-procedure tail utilities/recalls, decoder displacement and original-zero agreement. Every seed and eligible procedure is retained.",
            interpretation="Compare trained versus fixed decoder within each dictionary, then sparse versus ordinary. Gains in both dictionaries support task-dependent directions without sparse-specific superiority. A held gain without complete-video benefit identifies a remaining task proxy gap. No benefit limits this fixed-encoder decoder/gain recipe; it does not reject the whole SAE family. Generic supervised dictionary refinement is prior art.",
            references=["SRC-E2E-SAE-2024", "SRC-KL-FINETUNE-2025", "SRC-TASK-DL-2010", "SRC-SUPCON-NEURIPS2020"]))
        print(run, flush=True)
        return
    python = str(ROOT / "artifacts/environments/modern/Scripts/python.exe")
    target = ("\\\\?\\" if os.name == "nt" else "") + str(run.resolve())
    config = read_json(run / "config.json")
    stages = [dict(id="train", label="固定与可训练decoder的完整比较", kind="train",
        command=[python, "scripts/train_decoder_identity_tail.py", "--run", target, "--resume"],
        progress=str(run / "training_progress.json"), output=str(run / "training_summary.json"),
        units=48, unit="组模型", estimate_seconds=training_seconds, resources=[])]
    for seed in config["seeds"]:
        stages.append(dict(id=f"evaluate_{seed}", label=f"种子{seed}完整提示评价", kind="evaluate",
            command=[python, "scripts/evaluate_decoder_identity_tail.py", "--run", target, "--phase", "evaluate",
                     "--seed", str(seed), "--resume"],
            progress=str(run / "evaluation" / f"seed{seed}" / "progress.json"),
            output=str(run / "evaluation" / f"seed{seed}" / "summary.json"),
            units=10, unit="个procedure", estimate_seconds=420, resources=["disk-e-io", "disk-d-io"]))
    stages.append(dict(id="summary", label="全部模型和procedure结果图表", kind="summary",
        command=[python, "scripts/evaluate_decoder_identity_tail.py", "--run", target, "--phase", "summary", "--resume"],
        progress=str(run / "summary_progress.json"), output=str(run / "figures/manifest.json"),
        units=1, unit="组结果", estimate_seconds=20, resources=[]))
    path = run / "batch_plan.json"
    if path.exists():
        raise FileExistsError(path)
    files = ["scripts/train_decoder_identity_tail.py", "scripts/prepare_decoder_identity_batch.py",
             "scripts/evaluate_decoder_identity_tail.py", "scripts/run_sae_acknowledgement_batch.py"]
    atomic_write_json(path, dict(question=read_json(run / "protocol.json")["question"], stages=stages,
        config_sha256=file_sha256(run / "config.json"), protocol_sha256=file_sha256(run / "protocol.json"),
        source_hashes={name: file_sha256(ROOT / name) for name in files},
        estimate_basis="Real eight-condition fitting smoke and recent complete-video throughput."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("config", "plan"), required=True)
    parser.add_argument("--training-seconds", type=float, default=3600.)
    args = parser.parse_args()
    prepare(args.run, args.phase, args.training_seconds)
