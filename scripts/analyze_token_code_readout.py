import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import train_acknowledgement_sae as shared
from train_temporal_shared_sae import protected_statistics
from src.checkpoint_io import atomic_write_json, read_json


def analyze(source, output, limit):
    output.mkdir(parents=True, exist_ok=True)
    training = read_json(source / "training_summary.json")
    records = read_json(source / "records.json")
    selection = {(r["method"], r["seed"]): r["step"] for r in training["selection"]}
    rows, identities, checked = [], [], 0
    jobs = training["outputs"][:limit] if limit else training["outputs"]
    for job in jobs:
        step = selection[job["method"], job["seed"]]
        path = Path(job["directory"]) / f"held_{step:04d}.npz"
        with np.load(path, allow_pickle=False) as data:
            indices, a, b = data["indices"], data["source"], data["query"]
            embeddings, pooled, saved = data["embeddings"], data["pooled_codes"], data["scores"]
            labels, cross = data["same_identity"], data["cross_interval"]
        videos = sorted({records[int(i)]["video_id"] for i in indices})
        pairs = shared.chronological_pairs(records, videos)
        np.testing.assert_array_equal(a, [p["source_index"] for p in pairs])
        np.testing.assert_array_equal(b, [p["query_index"] for p in pairs])
        np.testing.assert_array_equal(labels, [p["same_identity"] for p in pairs])
        position = {int(index): j for j, index in enumerate(indices)}
        left = np.array([position[int(i)] for i in a])
        right = np.array([position[int(i)] for i in b])
        reproduction = np.sum(embeddings[left].astype(float) * embeddings[right], axis=1)
        np.testing.assert_allclose(reproduction, saved, atol=2e-7, rtol=0)
        scores = {"learned": saved}
        if pooled.shape[1]:
            if np.any(pooled < 0) or not np.isfinite(pooled).all() or np.any(pooled.sum(1) <= 0):
                raise ValueError("Invalid nonnegative code")
            code = pooled.astype(np.float64)
            unit = code / np.linalg.norm(code, axis=1, keepdims=True)
            distribution = code / code.sum(axis=1, keepdims=True)
            root = np.sqrt(distribution)
            scores["code_cosine"] = np.sum(unit[left] * unit[right], axis=1)
            scores["hellinger"] = np.sum(root[left] * root[right], axis=1)
            scores["intersection"] = np.minimum(distribution[left], distribution[right]).sum(1)
            if any(np.any((v < -1e-10) | (v > 1 + 1e-10)) for k, v in scores.items() if k != "learned"):
                raise ValueError("Invalid histogram similarity")
        identity = dict(method=job["method"], seed=job["seed"], fold=job["fold"], step=step,
                        partition="validation" if job["fold"] == "full" else "training_oof")
        for name, values in scores.items():
            report = protected_statistics(pairs, values, .99)
            for video, row in report["by_video"].items():
                mask = np.array([p["video_id"] == video for p in pairs])
                rows.append(dict(**identity, readout=name, video_id=video, **row,
                    cross_interval_recall=float(np.mean(values[mask & cross] > report["threshold"])) if np.any(mask & cross) else None,
                    auroc=float(roc_auc_score(labels[mask], values[mask])), threshold=report["threshold"]))
        shared.save_npz(output / f"{job['method']}_{job['seed']}_{job['fold']}.npz", **scores,
                        source=a, query=b, same_identity=labels, cross_interval=cross)
        identities.append(dict(path=str(path), sha256=shared.file_sha256(path)))
        checked += 1
        print("CODE_READOUT", checked, "/", len(jobs), job["method"], job["fold"], flush=True)
    with (output / "procedures.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for partition, method, readout in sorted({(r["partition"], r["method"], r["readout"]) for r in rows}):
        group = [r for r in rows if (r["partition"], r["method"], r["readout"]) == (partition, method, readout)]
        summary.append(dict(partition=partition, method=method, readout=readout,
            procedures=len({r["video_id"] for r in group}),
            **{key: float(np.mean([r[key] for r in group if r[key] is not None]))
               for key in ("recall", "negative_retention", "cross_interval_recall", "auroc")}))
    atomic_write_json(output / "summary.json", dict(status="COMPLETE", results=summary,
        inputs=identities, code_sha256=shared.file_sha256(Path(__file__)), completed_at=shared.now(),
        scope="Post-hoc frozen-code readout diagnosis; original selected checkpoints; no new fitting or independent confirmation."))
    for row in summary:
        print(row["partition"], row["method"], row["readout"],
              f"recall={row['recall']:.4f} cross={row['cross_interval_recall']:.4f} AUROC={row['auroc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    analyze(args.source, args.output, args.limit)
