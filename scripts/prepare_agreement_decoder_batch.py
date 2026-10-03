import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import prepare_decoder_identity_batch as previous
from src.checkpoint_io import atomic_write_json, read_json


def prepare(run):
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    parent = ROOT / "results/runs/20261003T1145Z_decoder_identity_tail_v1"
    config = read_json(parent / "config.json")
    config.update(run_id=run.name, run_dir=str(run), gradient_rule="agreement", mean_gradient_run=str(parent))
    atomic_write_json(run / "config.json", config)
    atomic_write_json(run / "protocol.json", dict(
        question="Do gradient directions shared by the three existing training partitions improve procedure transfer and protected prompt removal?",
        motivation="The completed decoder experiment drives fitted tail loss near zero. Its saved-model transfer experiment shows a large fit/held gap even with unchanged training heads, and a smaller additional evaluation-head decrement.",
        hypothesis="Suppressing conflicting partition gradients constrains updates toward shared task effects and may reduce procedure-specific fitting.",
        algorithm="Use the weighted gradient from each existing crossfit-head partition separately. For every parameter coordinate, retain their sum only if all three signs are strictly positive or all strictly negative; otherwise set the combined gradient to zero. Apply the existing gradient-norm clipping and Adam step to these gradients. Decoder unit-column normalization remains unchanged. Zero coordinates contribute zero. All coefficients, optimizer state initialization,500 steps, loss terms and observations equal the completed mean-gradient run.",
        comparisons="48 fits: sparse/ordinary dictionary, fixed/trainable decoder, three seeds, three outer-held scopes plus full training. Compare all four outcomes with the saved matched mean-gradient experiment, original head, historical global SAE and three norm-matched random controls in complete-video evaluation. No model selection.",
        data="Existing19 training procedures and original grouped exposure; no reserved independent data. The three gradient groups are existing crossfit partitions and use different heads. They are not claimed to identify acquisition domains. Each head excludes its assigned fitting partition and all outer-held procedures.",
        outcomes="Primary: full-procedure repeat removal at the unchanged common protection floor and first-prompt requirement. Report extension with unchanged development thresholds, all seeds/procedures, held utility/recall, training loss and per-parameter fraction/norm of retained gradients. Compare agreement-minus-mean effects within each dictionary/decoder arm.",
        decision="Protected application and held improvement would support further confirmation of shared-update training. Fitted improvement alone supplies no evidence of useful transfer. Absent usable updates causes an explicit failure with the last valid checkpoint. A negative result restricts this Agr-Sum configuration, not the SAE family. No arbitrary coefficient or agreement-threshold search is planned.",
        budget="CPU one thread, no GPU.48 full-batch500-step jobs and three full evaluations. Normal E then D leases during video evaluation. Existing checkpoints, coordinator and monitor; duration estimated from actual smoke and recent full throughput.",
        prior_art="Agr-Sum is adopted from SRC-DGVGS-2021; task-directed dictionary training and residual edits retain existing attributions. No novelty claim for gradient consensus itself."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--phase", choices=("config", "plan"), required=True)
    parser.add_argument("--training-seconds", type=float, default=3600.)
    args = parser.parse_args()
    if args.phase == "config":
        prepare(args.run)
    else:
        previous.prepare(args.run, "plan", args.training_seconds)
