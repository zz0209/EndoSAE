"""Supervised REAL-Colon pilot: prepare inputs, LR1 tokens, and task comparisons.

The persistent consumer is the sampled clip manifest in the current task run.
Extraction stays in build_realcolon_visibility_pack; optimization and evaluation
live here rather than in the historical plotting/identity runner. No image,
checkpoint, clinical label, or test-set selection is performed by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def project_boxes(frame):
    """Weak patch-center box membership, with a small-box overlap fallback.

    Coordinates are x,y throughout. No full-image box is silently retained after
    being cropped out. This target is bounding-box support, not segmentation.
    """
    w, h = frame["width"], frame["height"]
    if w <= 0 or h <= 0:
        raise ValueError("invalid image dimensions")
    yy, xx = np.meshgrid(np.arange(14), np.arange(14), indexing="ij")
    centers_x, centers_y = (xx.ravel() + 0.5) * 16, (yy.ravel() + 0.5) * 16
    mask = np.zeros(196, dtype=bool)
    fallbacks = 0
    for record in frame["boxes_xyxy"]:
        x0, y0, x1, y1 = record["box"]
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError("invalid box coordinates")
        x0, x1 = max(0., x0 * 280. / w - 28.), min(224., x1 * 280. / w - 28.)
        y0, y1 = max(0., y0 * 224. / h), min(224., y1 * 224. / h)
        if x1 <= x0 or y1 <= y0:
            continue
        inside = (centers_x >= x0) & (centers_x < x1) & (centers_y >= y0) & (centers_y < y1)
        if not inside.any():
            widths = np.maximum(0., np.minimum(centers_x + 8, x1) - np.maximum(centers_x - 8, x0))
            heights = np.maximum(0., np.minimum(centers_y + 8, y1) - np.maximum(centers_y - 8, y0))
            overlap = widths * heights
            if overlap.max() <= 0:
                raise RuntimeError("visible box has no intersecting patch")
            inside[int(overlap.argmax())] = True
            fallbacks += 1
        mask |= inside
    return mask, fallbacks


def appearance_features(array):
    """RGB means/std, simple texture/specularity, and position; no box input."""
    patch = array.reshape(14, 16, 14, 16, 3).transpose(0, 2, 1, 3, 4).reshape(196, 16, 16, 3)
    means, std = patch.mean((1, 2)), patch.std((1, 2))
    bright = patch.max(-1)
    saturation = patch.max(-1) - patch.min(-1)
    extra = np.stack([patch.mean(-1).std((1, 2)),
                      ((bright > .8) & (saturation < .12)).mean((1, 2)),
                      (patch[..., 1] - .5 * (patch[..., 0] + patch[..., 2])).mean((1, 2))], axis=1)
    yy, xx = np.meshgrid(np.linspace(-1, 1, 14), np.linspace(-1, 1, 14), indexing="ij")
    base = np.concatenate([means, std, extra, xx.reshape(-1, 1), yy.reshape(-1, 1)], axis=1)
    products = np.stack([base[:, i] * base[:, j] for i in range(base.shape[1])
                         for j in range(i, base.shape[1])], axis=1)
    return np.concatenate([base, products], axis=1).astype(np.float32)


def tokens_to_frames(tokens):
    if tokens.shape != (1, 1569, 768):
        raise ValueError("expected global CLS plus spatial-major/time-minor tokens")
    return tokens[0, 1:].reshape(196, 8, 768).transpose(1, 0, 2)


def select(run, config, _rows):
    """Apply the predeclared cohort and pilot sampling rule to official XML."""
    import tarfile
    from scripts.audit_realcolon_visibility import parse_xml, FRAME_RE
    target = run / "clip_manifest.jsonl"
    if target.exists():
        raise RuntimeError("refusing to overwrite selected clips")
    clips, reports = [], []
    for spec in config["selected_videos"]:
        video = spec["video_id"]
        path = Path(config["annotation_directory"]) / (video + "_annotations.tar.gz")
        frames, all_ids, excluded = {}, set(), []
        with tarfile.open(str(path), "r:gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".xml"):
                    continue
                match = FRAME_RE.search(member.name)
                if not match:
                    raise RuntimeError("unrecognized annotation member")
                index = int(match.group(1))
                if index in all_ids:
                    raise RuntimeError("duplicate XML frame identity")
                all_ids.add(index)
                try:
                    w, h, boxes, anomalies, _ = parse_xml(archive.extractfile(member).read())
                except (ValueError, TypeError) as error:
                    excluded.append({"frame": index, "reason": str(error)})
                    continue
                if anomalies:
                    excluded.append({"frame": index, "reason": "existing parser flagged annotation anomaly"})
                    continue
                frame = {"frame_index": index, "xml_member": member.name, "width": w, "height": h,
                         "boxes_xyxy": [{"lesion_id": uid, "box": list(box)} for uid, box in boxes]}
                mask, _ = project_boxes(frame)
                frame["class"] = 1 if mask.any() else (0 if not boxes else -1)
                frames[index] = frame
        pools = {0: [], 1: []}
        for first in range(min(all_ids), max(all_ids) - 6, 8):
            indices = list(range(first, first + 8))
            if not all(index in frames for index in indices):
                continue
            labels = {frames[index]["class"] for index in indices}
            if len(labels) == 1 and next(iter(labels)) in pools:
                pools[next(iter(labels))].append(indices)
        count = config["clips_per_class_per_video"]
        used = []
        for label, pool in pools.items():
            if len(pool) < count:
                raise RuntimeError("pre-pixel cohort eligibility failed: {0}, class{1}, blocks{2}".format(video, label, len(pool)))
            for i in range(count):
                indices = pool[i * (len(pool) - 1) // (count - 1)]
                clips.append({"clip_id": video + "_" + str(indices[0]), "video_id": video, "split": spec["split"],
                              "class": label, "frames": [frames[index] for index in indices]})
                used.extend(indices)
        if len(used) != len(set(used)):
            raise RuntimeError("overlapping selected clips")
        record = {"video_id": video, "xml_frames": len(all_ids), "xml_index_gaps": max(all_ids) - min(all_ids) + 1 - len(all_ids),
                  "excluded_frames": len(excluded), "eligible_positive_blocks": len(pools[1]),
                  "eligible_background_blocks": len(pools[0]), "annotation_sha256": digest(path)}
        reports.append(record)
        write_json(run / (video + "_annotation_exclusions.json"), excluded)
        print("SELECT " + json.dumps(record), flush=True)
    old_run = ROOT / config["reuse_pilot_run"]
    old_clips = {row["clip_id"]: row for line in (old_run / "clip_manifest.jsonl").read_text().splitlines() for row in [json.loads(line)]}
    new_clips = {row["clip_id"]: row for row in clips}
    if not all(new_clips.get(key) == row for key, row in old_clips.items()):
        raise RuntimeError("pilot clip reuse changed sampling or labels")
    target.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in clips), encoding="utf-8")
    write_json(run / "sampling_summary.json", {"status": "SELECTED_BEFORE_NEW_PIXELS", "n_clips": len(clips),
               "n_frames": len(clips) * 8, "per_video": reports, "pilot_clips_exactly_reused": len(old_clips),
               "manifest_sha256": digest(target), "cohort_config_sha256": digest(run / "cohort_config.json"),
               "runner_sha256": digest(__file__)})


def cache_reuse(config, rows, phase):
    """Bind identical clips to a verified earlier input/token cache."""
    if not config.get("reuse_source_run"):
        return {}, {}, None
    source = Path(config["reuse_source_run"])
    source_config = json.loads((source / "execution_config.json").read_text(encoding="utf-8"))
    for key in ("label_rule", "state_exchange_sha256"):
        if config[key] != source_config[key]:
            raise RuntimeError("cache reuse preprocessing/model identity mismatch")
    old_rows = [json.loads(line) for line in (source / "clip_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {row["clip_id"]: (i, row) for i, row in enumerate(old_rows)}
    if len(by_id) != len(old_rows) or len({row["clip_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate clip identity in cache reuse")
    mapping = {}
    for i, row in enumerate(rows):
        if row["clip_id"] in by_id:
            old_index, old_row = by_id[row["clip_id"]]
            if row != old_row:
                raise RuntimeError("reused clip frames, labels or split changed")
            mapping[i] = old_index
    source_prepared = json.loads((source / "prepared.json").read_text(encoding="utf-8"))
    if source_prepared["manifest_sha256"] != digest(source / "clip_manifest.jsonl"):
        raise RuntimeError("reuse source manifest changed")
    cache = Path(source_config["cache_dir"])
    if phase == "prepare":
        expected = source_prepared["assets"]
        names = ("inputs.npy", "masks.npy", "appearance.npy")
        stage = source / "prepared.json"
    elif phase == "encode":
        stage = source / "encoded.json"
        encoded = json.loads(stage.read_text(encoding="utf-8"))
        if encoded["manifest_sha256"] != source_prepared["manifest_sha256"] or encoded["prepared_sha256"] != digest(source / "prepared.json"):
            raise RuntimeError("reuse source token/preparation binding mismatch")
        if encoded["state_exchange_sha256"] != config["state_exchange_sha256"] or encoded["layer"] != "final_normalized_tokens":
            raise RuntimeError("reuse source token model/layer mismatch")
        expected = {"tokens.npy": encoded["tokens_sha256"]}
        names = ("tokens.npy", "cls.npy")
    else:
        raise ValueError("unknown cache reuse phase")
    arrays, hashes = {}, {}
    for name in names:
        hashes[name] = digest(cache / name)
        if name in expected and hashes[name] != expected[name]:
            raise RuntimeError("reuse source cache bytes changed")
        arrays[name] = np.load(str(cache / name), mmap_mode="r", allow_pickle=False)
        if len(arrays[name]) != len(old_rows):
            raise RuntimeError("reuse source cache length mismatch")
    provenance = {"run": str(source), "source_stage_sha256": digest(stage), "assets": hashes,
                  "n_reused_clips": len(mapping), "mapping": mapping}
    return mapping, arrays, provenance


def prepare(run, config, rows):
    from PIL import Image
    receipt = json.loads((run / "extraction_summary.json").read_text())
    if receipt["manifest_sha256"] != digest(run / "clip_manifest.jsonl"):
        raise RuntimeError("extraction/selection mismatch")
    if receipt["status"] != "EXTRACTED_DIMENSIONS_VERIFIED" or receipt["frames"] != len(rows) * 8:
        raise RuntimeError("complete extraction required before preparation")
    reuse_mapping, reuse_arrays, reuse_provenance = cache_reuse(config, rows, "prepare")
    cache = Path(config["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "inputs.npy").exists() or (run / "prepared.json").exists():
        raise RuntimeError("prepared inputs already exist")
    frames_root = Path(receipt["output_dir"])
    inputs = np.lib.format.open_memmap(str(cache / "inputs.npy"), mode="w+", dtype="float32",
                                      shape=(len(rows), 3, 8, 224, 224))
    masks, appearance, fallbacks = [], [], 0
    mean = np.array([.485, .456, .406], dtype=np.float32)
    std = np.array([.229, .224, .225], dtype=np.float32)
    source_hashes = {}
    for filename in receipt["receipts"]:
        video = json.loads((run / filename).read_text())
        source_hashes.update({r["relative_path"]: r["sha256"] for r in video["frames"]})
    for i, row in enumerate(rows):
        if i in reuse_mapping:
            old_index = reuse_mapping[i]
            inputs[i] = reuse_arrays["inputs.npy"][old_index]
            masks.append(np.array(reuse_arrays["masks.npy"][old_index], copy=True))
            appearance.append(np.array(reuse_arrays["appearance.npy"][old_index], copy=True))
            fallbacks += sum(project_boxes(frame)[1] for frame in row["frames"])
            continue
        clip_masks, clip_appearance = [], []
        if len(row["frames"]) != 8:
            raise RuntimeError("expected eight frames")
        for t, frame in enumerate(row["frames"]):
            relative = row["video_id"] + "/{0:06d}.jpg".format(frame["frame_index"])
            path = frames_root / relative
            if digest(path) != source_hashes[relative]:
                raise RuntimeError("extracted frame changed")
            with Image.open(path) as image:
                if image.size != (frame["width"], frame["height"]):
                    raise RuntimeError("JPEG/XML dimension mismatch")
                crop = image.convert("RGB").resize((280, 224), Image.Resampling.BICUBIC).crop((28, 0, 252, 224))
                array = np.asarray(crop, dtype=np.float32) / 255.
            inputs[i, :, t] = ((array - mean) / std).transpose(2, 0, 1)
            mask, count = project_boxes(frame)
            if bool(mask.any()) != bool(row["class"]):
                raise RuntimeError("selected class contradicts visible crop support")
            clip_masks.append(mask)
            clip_appearance.append(appearance_features(array))
            fallbacks += count
        masks.append(clip_masks)
        appearance.append(clip_appearance)
        if i % 16 == 0:
            print("PREPARE {0}/{1}".format(i + 1, len(rows)), flush=True)
    inputs.flush()
    del inputs
    np.save(str(cache / "masks.npy"), np.asarray(masks, dtype=bool), allow_pickle=False)
    np.save(str(cache / "appearance.npy"), np.asarray(appearance, dtype=np.float32), allow_pickle=False)
    files = {name: digest(cache / name) for name in ("inputs.npy", "masks.npy", "appearance.npy")}
    write_json(run / "prepared.json", {"status": "PREPARED", "cache_dir": str(cache), "assets": files,
               "manifest_sha256": digest(run / "clip_manifest.jsonl"), "n_clips": len(rows),
               "small_box_fallbacks": fallbacks, "runner_sha256": digest(__file__), "cache_reuse": reuse_provenance})


def encode(run, config, rows):
    import torch
    if not torch.__version__.startswith("1.8.0") or sys.version_info[:2] != (3, 7):
        raise RuntimeError("encoding requires the existing LR1 Python3.7/torch1.8 CPU reference")
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_model_port_parity as parity
    torch.set_num_threads(config["cpu_threads"])
    torch.set_num_interop_threads(1)
    cache = Path(config["cache_dir"])
    prepared = json.loads((run / "prepared.json").read_text())
    if digest(cache / "inputs.npy") != prepared["assets"]["inputs.npy"]:
        raise RuntimeError("input exchange changed")
    if (cache / "tokens.npy").exists():
        raise RuntimeError("refusing to overwrite activation cache")
    reuse_mapping, reuse_arrays, reuse_provenance = cache_reuse(config, rows, "encode")
    state_path = ROOT / config["state_exchange"]
    if digest(state_path) != config["state_exchange_sha256"]:
        raise RuntimeError("state exchange mismatch")
    model = parity.build_model(torch, parity.load_timesformer(str(ROOT / "third_party/Endo-FM/models")))
    model.load_state_dict(parity.load_state_exchange(str(state_path), torch, np), strict=True)
    model.eval()
    inputs = np.load(str(cache / "inputs.npy"), mmap_mode="r", allow_pickle=False)
    outputs = np.lib.format.open_memmap(str(cache / "tokens.npy"), mode="w+", dtype="float32",
                                       shape=(len(rows), 8, 196, 768))
    cls_values = []
    started = time.perf_counter()
    interface = {}
    for i in range(len(rows)):
        if i in reuse_mapping:
            old_index = reuse_mapping[i]
            outputs[i] = reuse_arrays["tokens.npy"][old_index]
            cls_values.append(np.array(reuse_arrays["cls.npy"][old_index], copy=True))
            continue
        value = torch.from_numpy(np.array(inputs[i:i + 1], copy=True))
        with torch.no_grad():
            tokens = model.forward_features(value, get_all=True)
            if tuple(tokens.shape) != (1, 1569, 768):
                raise RuntimeError("final token shape mismatch")
            if not interface:
                baseline = model(value)
                interface["all_tokens_cls_exactly_matches_standard_forward"] = bool(torch.equal(baseline, tokens[:, 0]))
                if not interface["all_tokens_cls_exactly_matches_standard_forward"]:
                    raise RuntimeError("all-token interface changed ordinary model output")
        array = tokens.numpy()
        if not np.isfinite(array).all():
            raise RuntimeError("nonfinite model tokens")
        # G1 coordinate fixture: spatial-major/time-minor, global CLS at index 0.
        outputs[i] = tokens_to_frames(array)
        cls_values.append(array[0, 0])
        if i % 8 == 0 or i + 1 == len(rows):
            print("ENCODE {0}/{1} elapsed={2:.1f}s".format(i + 1, len(rows), time.perf_counter() - started), flush=True)
    outputs.flush()
    del outputs
    np.save(str(cache / "cls.npy"), np.asarray(cls_values), allow_pickle=False)
    write_json(run / "encoded.json", {"status": "ENCODED_LR1", "layer": "final_normalized_tokens",
               "layout": "clip,time,spatial,channel", "shape": [len(rows), 8, 196, 768],
               "tokens_sha256": digest(cache / "tokens.npy"), "prepared_sha256": digest(run / "prepared.json"),
               "manifest_sha256": digest(run / "clip_manifest.jsonl"), "state_exchange_sha256": digest(state_path),
               "interface": interface, "python": platform.python_version(), "torch": torch.__version__,
               "numpy": np.__version__, "device": "cpu", "elapsed_seconds": time.perf_counter() - started,
               "runner_sha256": digest(__file__), "cache_reuse": reuse_provenance,
               "n_newly_encoded_clips": len(rows) - len(reuse_mapping)})


def binary_metrics(labels, scores):
    """ROC AUC and non-interpolated AP with exact tie grouping."""
    y, s = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    if y.shape != s.shape or not np.isfinite(s).all() or not y.any() or y.all():
        raise ValueError("binary metrics need finite, mixed-class aligned arrays")
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    tp = np.cumsum(ys)[ends].astype(float)
    fp = (ends + 1) - tp
    recall, fpr = tp / y.sum(), fp / (~y).sum()
    auc = np.trapezoid(np.r_[0., recall], np.r_[0., fpr]) if hasattr(np, "trapezoid") else np.trapz(np.r_[0., recall], np.r_[0., fpr])
    ap = np.sum(np.diff(np.r_[0., recall]) * tp / (ends + 1))
    return {"auroc": float(auc), "average_precision": float(ap)}


def summarize(predictions, masks, rows):
    frame_scores = np.sort(predictions, axis=-1)[..., -4:].mean(-1)
    clip_scores = frame_scores.mean(-1)
    labels = np.array([row["class"] for row in rows], dtype=bool)
    train = np.array([row["split"] == "train" for row in rows])
    threshold = float(np.quantile(clip_scores[train & ~labels], .95, method="higher"))
    result = {"train_negative_95th_percentile_threshold": threshold, "per_video": {}}
    for video in sorted({row["video_id"] for row in rows}):
        selected = np.array([row["video_id"] == video for row in rows])
        pos = selected & labels
        neg = selected & ~labels
        ap = [binary_metrics(m, p)["average_precision"] for m, p in zip(masks[pos].reshape(-1, 196), predictions[pos].reshape(-1, 196)) if m.any() and not m.all()]
        peak = predictions[pos].argmax(-1)
        hits = np.take_along_axis(masks[pos], peak[..., None], axis=-1)[..., 0]
        result["per_video"][video] = {
            "split": rows[int(np.flatnonzero(selected)[0])]["split"],
            "clip_detection": binary_metrics(labels[selected], clip_scores[selected]),
            "frame_detection": binary_metrics(np.repeat(labels[selected], 8), frame_scores[selected].reshape(-1)),
            "positive_frame_mean_patch_ap": float(np.mean(ap)) if ap else None,
            "positive_frame_pointing_hit_rate": float(hits.mean()),
            "clip_sensitivity_at_train_threshold": float((clip_scores[pos] >= threshold).mean()),
            "background_clip_fpr_at_train_threshold": float((clip_scores[neg] >= threshold).mean()),
            "n_clips": int(selected.sum()), "n_positive_clips": int(pos.sum()),
        }
    return result


def fit_head(features, train_mask, labels, config, output, name, torch):
    """Fixed-budget shared logistic head; all fitted statistics are train-only."""
    torch.manual_seed(config["seed"])
    train_x, train_y = features[train_mask], labels[train_mask]
    mean = train_x.mean(0)
    scale = train_x.std(0, unbiased=False).clamp_min(1e-4)
    if config.get("memory_efficient_head", False):
        # Boolean indexing above made a private copy; preserve input features.
        train_x.sub_(mean).div_(scale)
    else:
        train_x = (train_x - mean) / scale
    head = torch.nn.Linear(features.shape[1], 1).to(features.device)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config["head_learning_rate"], weight_decay=config["head_weight_decay"])
    positive_weight = (1. - train_y.mean()) / train_y.mean()
    generator = torch.Generator(device=features.device).manual_seed(config["seed"])
    history = []
    for step in range(config["head_steps"]):
        index = torch.randint(len(train_x), (config["head_batch_size"],), generator=generator, device=features.device)
        logits = head(train_x[index]).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_y[index], pos_weight=positive_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite head loss")
        if step % 200 == 0 or step + 1 == config["head_steps"]:
            history.append({"step": step + 1, "minibatch_loss": float(loss.detach())})
    predictions = []
    with torch.no_grad():
        for batch in features.split(4096):
            predictions.append(torch.sigmoid(head((batch - mean) / scale)).squeeze(-1).cpu().numpy())
    np.savez(output / (name + "_head.npz"), weight=head.weight.detach().cpu().numpy(), bias=head.bias.detach().cpu().numpy(),
             mean=mean.cpu().numpy(), scale=scale.cpu().numpy())
    write_json(output / (name + "_training.json"), {"history": history, "positive_weight": float(positive_weight),
               "n_training_tokens": len(train_x), "n_parameters": sum(p.numel() for p in head.parameters())})
    return np.concatenate(predictions)


def fit(run, config, rows):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from src.sae.baselines import TopKAutoencoder
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for fixed training execution")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    cache = Path(config["cache_dir"])
    encoded = json.loads((run / "encoded.json").read_text())
    if encoded["manifest_sha256"] != digest(run / "clip_manifest.jsonl"):
        raise RuntimeError("encoded/selected manifest mismatch")
    if digest(cache / "tokens.npy") != encoded["tokens_sha256"]:
        raise RuntimeError("token asset changed")
    output = run / "comparison_v0"
    output.mkdir(exist_ok=False)
    write_json(output / "config.json", config)
    (output / "runner_snapshot.py").write_bytes(Path(__file__).read_bytes())
    started = time.perf_counter()
    masks = np.load(cache / "masks.npy", allow_pickle=False)
    n = len(rows)
    train_clips = np.array([row["split"] == "train" for row in rows])
    train_mask = torch.tensor(np.repeat(train_clips, 8 * 196), device=device)
    labels = torch.tensor(masks.reshape(-1).astype(np.float32), device=device)
    raw = torch.tensor(np.load(cache / "tokens.npy", allow_pickle=False).reshape(-1, 768), device=device)
    mean = raw[train_mask].mean(0)
    rms = ((raw[train_mask] - mean).square().mean()).sqrt().clamp_min(1e-6)
    x = (raw - mean) / rms
    del raw
    all_predictions, summaries = {}, {}

    def compare(name, features):
        head_config = dict(config, head_weight_decay=config.get("method_head_weight_decay", {}).get(name, config["head_weight_decay"]))
        prediction = fit_head(features, train_mask, labels, head_config, output, name, torch).reshape(n, 8, 196)
        all_predictions[name] = prediction
        summaries[name] = summarize(prediction, masks, rows)
        write_json(output / (name + "_metrics.json"), summaries[name])
        np.save(output / (name + "_predictions.npy"), prediction, allow_pickle=False)
        print("FIT {0} {1}".format(name, json.dumps({v: s for v, s in summaries[name]["per_video"].items() if s["split"] != "train"})), flush=True)

    compare("raw_endofm", x)
    appearance = torch.tensor(np.load(cache / "appearance.npy", allow_pickle=False).reshape(n * 8 * 196, -1), device=device)
    compare("appearance_position_quadratic", appearance)
    del appearance
    train_x = x[train_mask]
    # Exact 768x768 training covariance; no test or development fitting.
    with torch.no_grad():
        covariance = train_x.T @ train_x / len(train_x)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors[:, -64:].flip(1)
        np.savez(output / "pca.npz", mean=mean.cpu().numpy(), rms=rms.cpu().numpy(),
                 basis=basis.cpu().numpy(), eigenvalues=eigenvalues.cpu().numpy())
    for rank in (32, 64):
        compare("pca{0}".format(rank), x @ basis[:, :rank])

    torch.manual_seed(config["seed"])
    sae = TopKAutoencoder(768, config["topk_width"], config["topk_k"]).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config["topk_learning_rate"])
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    history = []
    for step in range(config["topk_steps"]):
        index = torch.randint(len(train_x), (config["topk_batch_size"],), generator=generator, device=device)
        batch = train_x[index]
        reconstruction, code = sae(batch, inference=False)
        loss = (reconstruction - batch).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        sae.project_decoder_gradient_()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.)
        optimizer.step()
        sae.normalize_decoder_()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite SAE loss")
        if step % 250 == 0 or step + 1 == config["topk_steps"]:
            history.append({"step": step + 1, "mse": float(loss.detach())})
            print("TOPK {0}/{1} mse={2:.4f}".format(step + 1, config["topk_steps"], float(loss.detach())), flush=True)
    codes = torch.empty((len(x), config["topk_width"]), device=device)
    squared_error = torch.zeros(len(x), device=device)
    with torch.no_grad():
        for offset in range(0, len(x), 4096):
            batch = x[offset:offset + 4096]
            reconstruction, code = sae(batch, inference=True)
            codes[offset:offset + len(batch)] = code
            squared_error[offset:offset + len(batch)] = (reconstruction - batch).square().sum(1)
    np.savez(output / "topk_sae.npz", **{key: value.detach().cpu().numpy() for key, value in sae.state_dict().items()},
             input_mean=mean.cpu().numpy(), input_rms=rms.cpu().numpy())
    sae_report = {"history": history, "width": config["topk_width"], "k_slots": config["topk_k"],
                  "fit_scope": sorted({row["video_id"] for row in rows if row["split"] == "train"}), "train_dead_fraction": float((codes[train_mask].sum(0) == 0).float().mean()),
                  "training_l0": float((codes[train_mask] > 0).float().sum(1).mean()),
                  "development_l0": float((codes[~train_mask] > 0).float().sum(1).mean()),
                  "train_nmse": float(squared_error[train_mask].sum() / train_x.square().sum()),
                  "development_nmse": float(squared_error[~train_mask].sum() / x[~train_mask].square().sum())}
    write_json(output / "topk_reconstruction.json", sae_report)
    if config.get("memory_efficient_head", False):
        del train_x, x, squared_error, sae, optimizer, batch, reconstruction, code
    compare("topk32_target_domain", codes)
    report = {"status": config.get("completion_status", "COMPLETED_DEVELOPMENT_PILOT"), "methods": summaries, "topk_reconstruction": sae_report,
              "manifest_sha256": digest(run / "clip_manifest.jsonl"), "encoded_sha256": digest(run / "encoded.json"),
              "elapsed_seconds": time.perf_counter() - started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
              "torch": torch.__version__, "python": platform.python_version(), "numpy": np.__version__,
              "limitations": config.get("limitations", ["one exposed development video, no independent confirmation", "training metrics are in-sample",
                              "box-support labels are not lesion segmentation", "class-balanced sparse temporal sample",
                              "fixed exploratory budgets are not tuned strong-baseline ceilings", "no causal or clinical safety claim"])}
    write_json(output / "summary.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["select", "prepare", "encode", "fit"])
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config_file = "cohort_config.json" if args.phase == "select" else "execution_config.json"
    config = json.loads((run / config_file).read_text(encoding="utf-8"))
    rows = [] if args.phase == "select" else [json.loads(line) for line in (run / "clip_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    write_json(run / "execution_status.json", {"phase": args.phase, "status": "RUNNING", "pid": os.getpid(),
               "started_at_unix": time.time(), "runner_sha256": digest(__file__)})
    try:
        globals()[args.phase](run, config, rows)
    except Exception as error:
        write_json(run / "execution_status.json", {"phase": args.phase, "status": "FAILED", "error": repr(error), "ended_at_unix": time.time()})
        raise
    write_json(run / "execution_status.json", {"phase": args.phase, "status": "COMPLETED", "ended_at_unix": time.time()})


if __name__ == "__main__":
    main()
