import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_decoder_identity_tail as training
from verify_decoder_identity_tail import compare_arrays
from src.checkpoint_io import atomic_write_json, read_json
from src.token_memory_edit import BoundedGains, FrozenSupCon


def verify(run):
    config = read_json(run / "config.json")
    output = run / "gradient_verification"
    output.mkdir()
    seed = config["seeds"][0]
    records = read_json(Path(config["parent_dictionary_run"]) / "descriptor_records.json")
    held = training.crossfit.outer_folds(config)[0]
    fit = [v for v in config["train_video_ids"] if v not in held]
    checks = []
    for method in config["methods"]:
        assets = training.parent.load_assets(config, records, method, seed, 0, fit, held)
        model = assets["model"]
        initial = model.decoder.weight.detach().clone()
        state = torch.load(run / "smoke/inner_folds" / method / f"seed{seed}/fold0/trainable_decoder/checkpoint.pt",
                           weights_only=True, map_location="cpu")
        model.load_state_dict(state["model"], strict=True)
        model.decoder.weight.requires_grad_(True)
        gains = BoundedGains(1024, config["gain_bound"])
        gains.load_state_dict(state["gains"], strict=True)
        parameters = list(gains.parameters()) + [model.decoder.weight]
        parts = [training.design(assets, records, plan["held_video_ids"], FrozenSupCon(plan["directory"]), True)
                 for plan in training.crossfit.head_plan(config, records, 0)]
        sizes = [len({row["video_id"] for row in part["rows"]}) for part in parts]
        gradients = [[] for _ in parameters]
        for part, size in zip(parts, sizes):
            terms = training.loss_terms(model, gains, torch.as_tensor(assets["scale"]), initial, part, config)
            loss = (terms[0] + config["identity_weight"] * terms[1] + config["decoder_anchor_weight"] * terms[2]) * size / sum(sizes)
            for storage, gradient in zip(gradients, torch.autograd.grad(loss, parameters)):
                storage.append(gradient.detach())
        for index, values in enumerate(gradients):
            actual, mask = training.agreement_gradient(values)
            array = np.stack([v.numpy() for v in values])
            independent_mask = np.logical_or(np.min(array, axis=0) > 0, np.max(array, axis=0) < 0)
            expected = np.sum(array, axis=0) * independent_mask
            np.testing.assert_array_equal(mask.numpy(), independent_mask)
            np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-6, atol=1e-10)
            if np.any(actual.numpy()[~independent_mask] != 0):
                raise ValueError("Conflicting gradient survived")
            checks.append(dict(method=method, parameter=index, agreement_fraction=float(independent_mask.mean()),
                               maximum_error=float(np.max(np.abs(actual.numpy() - expected)))))
    mean_config = dict(config, gradient_rule="mean")
    training.run(mean_config, output / "mean_replay", True, False, None)
    original = Path(config["mean_gradient_run"]) / "smoke"
    count = 0
    for job in read_json(output / "mean_replay/training_summary.json")["outputs"]:
        folder = Path(job["folder"])
        for name in ("model.npz", "gains.npz", "held.npz"):
            count += compare_arrays(folder / name, original / folder.relative_to(output / "mean_replay") / name)
    receipt = dict(status="PASS", real_gradient_checks=checks, mean_replay_arrays_exact=count)
    atomic_write_json(output / "verification.json", receipt)
    print(receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    verify(parser.parse_args().run)
