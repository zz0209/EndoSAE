"""Semantic tests for the new task's coordinates, metrics and fitting boundary."""
import tempfile
import unittest
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from src.evaluation.realcolon_task import binary_metrics, cache_reuse, digest, fit_head, project_boxes, tokens_to_frames
from src.evaluation.realcolon_task_cv import training_folds


class RealColonTaskTests(unittest.TestCase):
    def test_cache_reuse_preserves_order_and_rejects_label_or_byte_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            rows = [{"clip_id": "a", "split": "train", "class": 1}, {"clip_id": "b", "split": "dev", "class": 0}]
            manifest = source / "clip_manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(row) for row in rows))
            config = {"cache_dir": str(source), "label_rule": "fixture", "state_exchange_sha256": "fixture"}
            (source / "execution_config.json").write_text(json.dumps(config))
            assets = {}
            for name in ("inputs.npy", "masks.npy", "appearance.npy"):
                np.save(source / name, np.array([[1., 2.], [3., 4.]], dtype=np.float32))
                assets[name] = digest(source / name)
            (source / "prepared.json").write_text(json.dumps({"manifest_sha256": digest(manifest), "assets": assets}))
            current = dict(config, reuse_source_run=str(source))
            mapping, arrays, provenance = cache_reuse(current, list(reversed(rows)), "prepare")
            self.assertEqual(mapping, {0: 1, 1: 0})
            np.testing.assert_array_equal(arrays["inputs.npy"][mapping[0]], [3., 4.])
            self.assertEqual(provenance["n_reused_clips"], 2)
            del arrays
            with self.assertRaisesRegex(RuntimeError, "frames, labels or split changed"):
                cache_reuse(current, [dict(rows[0], **{"class": 0})], "prepare")
            np.save(source / "masks.npy", np.zeros((2, 2), dtype=np.float32))
            with self.assertRaisesRegex(RuntimeError, "cache bytes changed"):
                cache_reuse(current, rows, "prepare")

    def test_memory_efficient_head_matches_original_and_preserves_features(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(7)
        features = torch.randn(160, 24)
        original = features.clone()
        labels = (features[:, 0] > 0).float()
        train = torch.arange(len(features)) < 120
        config = {"seed": 7, "head_steps": 80, "head_batch_size": 48,
                  "head_learning_rate": .01, "head_weight_decay": 1.}
        with tempfile.TemporaryDirectory() as directory:
            a = fit_head(features, train, labels, config, Path(directory), "original", torch)
            b = fit_head(features, train, labels, dict(config, memory_efficient_head=True), Path(directory), "efficient", torch)
            np.testing.assert_array_equal(a, b)
            self.assertTrue(torch.equal(features, original))

    def test_extraction_reuses_identical_clips_and_defers_missing_archives(self):
        from PIL import Image
        from scripts.build_realcolon_visibility_pack import extract_task_manifest
        image = io.BytesIO()
        Image.new("RGB", (12, 10), "red").save(image, format="JPEG")
        content = image.getvalue()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archives = root / "archives"
            archives.mkdir()
            def clip(video):
                return {"video_id": video, "split": "train", "class": 0,
                        "frames": [{"frame_index": 0, "width": 12, "height": 10, "boxes_xyxy": []}]}
            def archive(video):
                with tarfile.open(archives / (video + "_frames.tar.gz"), "w:gz") as stream:
                    member = tarfile.TarInfo(video + "_frames/" + video + "_0.jpg")
                    member.size = len(content)
                    stream.addfile(member, io.BytesIO(content))
            old = root / "old"
            old.mkdir()
            (old / "clip_manifest.jsonl").write_text(json.dumps(clip("001-001")) + "\n")
            archive("001-001")
            extract_task_manifest(old / "clip_manifest.jsonl", archives, old / "frames", old)
            # Source archive unavailable: reuse must consume saved frames, not scan it.
            (archives / "001-001_frames.tar.gz").unlink()
            new = root / "new"
            new.mkdir()
            manifest = new / "clip_manifest.jsonl"
            manifest.write_text("\n".join(json.dumps(clip(v)) for v in ["001-001", "002-001"]) + "\n")
            extract_task_manifest(manifest, archives, new / "frames", new, old, True)
            summary = json.loads((new / "extraction_summary.json").read_text())
            self.assertEqual(summary["status"], "PARTIAL_WAITING_FOR_ARCHIVES")
            self.assertEqual(summary["pending_videos"], ["002-001"])
            self.assertEqual((new / "frames/001-001/000000.jpg").read_bytes(), content)
            archive("002-001")
            extract_task_manifest(manifest, archives, new / "frames", new, old, True)
            summary = json.loads((new / "extraction_summary.json").read_text())
            self.assertEqual(summary["frames"], 2)
            self.assertEqual(summary["status"], "EXTRACTED_DIMENSIONS_VERIFIED")
            drift = root / "drift.jsonl"
            changed = clip("001-001")
            changed["class"] = 1
            drift.write_text(json.dumps(changed) + "\n")
            with self.assertRaisesRegex(RuntimeError, "clips or labels differ"):
                extract_task_manifest(drift, archives, root / "badframes", root / "badrun", old, True)

    def test_cv_is_video_grouped_and_excludes_development(self):
        rows = [{"video_id": video, "split": split} for video, split in
                [("a", "train"), ("dev", "dev"), ("b", "train"), ("c", "train"), ("a", "train")]]
        folds = list(training_folds(rows))
        self.assertEqual([f[0] for f in folds], ["a", "b", "c"])
        for held_out, selected, fit in folds:
            self.assertEqual(selected, [0, 2, 3, 4])
            self.assertTrue(all(rows[i]["video_id"] != held_out for i, use in zip(selected, fit) if use))
            self.assertTrue(all(rows[i]["video_id"] == held_out for i, use in zip(selected, fit) if not use))
        with self.assertRaises(ValueError):
            list(training_folds(rows[:3]))

    def frame(self, box):
        return {"width": 1120, "height": 896, "boxes_xyxy": [{"box": box}]}

    def test_xy_projection_on_asymmetric_image(self):
        mask, fallback = project_boxes(self.frame([688, 128, 752, 192]))
        self.assertEqual(np.flatnonzero(mask).tolist(), [2 * 14 + 9])
        self.assertEqual(fallback, 0)

    def test_small_box_and_cropped_out_box(self):
        mask, fallback = project_boxes(self.frame([723, 163, 725, 165]))
        self.assertEqual(np.flatnonzero(mask).tolist(), [2 * 14 + 9])
        self.assertEqual(fallback, 1)
        mask, fallback = project_boxes(self.frame([0, 128, 20, 192]))
        self.assertFalse(mask.any())
        self.assertEqual(fallback, 0)

    def test_no_annotation_is_not_created_by_projection(self):
        with self.assertRaises(KeyError):
            project_boxes({"width": 1120, "height": 896})
        mask, _ = project_boxes({"width": 1120, "height": 896, "boxes_xyxy": []})
        self.assertFalse(mask.any())

    def test_time_minor_exchange(self):
        tokens = np.zeros((1, 1569, 768), dtype=np.float32)
        tokens[:, 0] = -999
        for spatial in range(196):
            for frame in range(8):
                tokens[0, 1 + spatial * 8 + frame] = 1000 * frame + spatial
        frames = tokens_to_frames(tokens)
        self.assertEqual(frames.shape, (8, 196, 768))
        self.assertEqual(frames[6, 113, 7], 6113)
        self.assertFalse((frames == -999).any())

    def test_metrics_ties_and_direction(self):
        labels = np.array([0, 0, 1, 1])
        self.assertEqual(binary_metrics(labels, [0, 1, 2, 3]), {"auroc": 1., "average_precision": 1.})
        tied = binary_metrics(labels, [1, 1, 1, 1])
        self.assertEqual(tied, {"auroc": .5, "average_precision": .5})
        self.assertEqual(binary_metrics(labels, [3, 2, 1, 0])["auroc"], 0.)
        with self.assertRaises(ValueError):
            binary_metrics([0, 0], [1, 2])

    def test_head_learns_signal_and_never_fits_development_scale(self):
        import torch
        torch.set_num_threads(1)
        features = torch.tensor([[-1., 0.], [1., 0.]] * 64 + [[-1., 9999.], [1., 9999.]] * 16)
        labels = (features[:, 0] > 0).float()
        train = torch.arange(len(features)) < 128
        config = {"seed": 5, "head_steps": 150, "head_batch_size": 64,
                  "head_learning_rate": .05, "head_weight_decay": 0.}
        with tempfile.TemporaryDirectory() as directory:
            prediction = fit_head(features, train, labels, config, Path(directory), "signal", torch)
            result = binary_metrics(labels[~train].numpy(), prediction[~train.numpy()])
            self.assertEqual(result["auroc"], 1.)
            with np.load(Path(directory) / "signal_head.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["mean"], [0., 0.])
                self.assertLess(saved["scale"][1], .001)


if __name__ == "__main__":
    unittest.main()
