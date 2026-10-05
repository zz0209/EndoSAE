from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import tarfile
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_realcolon_visibility import FRAME_RE, parse_xml, safe_member
from src.evaluation.realcolon_task import digest, project_boxes, write_json


def candidate_blocks(indices, length):
    ordered = sorted(indices)
    starts, tails = [], 0
    run_start = 0
    for stop in range(1, len(ordered) + 1):
        if stop < len(ordered) and ordered[stop] == ordered[stop - 1] + 1:
            continue
        run = ordered[run_start:stop]
        starts.extend(run[offset] for offset in range(0, len(run) - length + 1, length))
        tails += len(run) % length
        run_start = stop
    return starts, tails


def temporal_selection(starts, requested):
    if len(starts) <= requested:
        return list(starts)
    targets = np.linspace(starts[0], starts[-1], requested)
    selected, lower = [], 0
    for position, target in enumerate(targets):
        upper = len(starts) - (requested - position)
        index = min(range(lower, upper + 1), key=lambda j: (abs(starts[j] - target), starts[j]))
        selected.append(starts[index])
        lower = index + 1
    return selected


def official_cohort(config):
    cohort, sources = [], {}
    known_videos, known_lesions = set(), set()
    for split in ("train", "val"):
        path = Path(config["official_instances"][split])
        data = json.loads(path.read_text(encoding="utf-8"))
        identities = sorted({row["identity_id"] for row in data["annotations"]})
        videos = defaultdict(list)
        for identity in identities:
            videos[identity.rsplit("_", 1)[0]].append(identity)
        if known_videos.intersection(videos) or known_lesions.intersection(identities):
            raise ValueError("official train and validation identities overlap")
        known_videos.update(videos)
        known_lesions.update(identities)
        if len(videos) != config["expected_official_videos"][split]:
            raise ValueError("unexpected official video count")
        sources[split] = {"path": str(path), "sha256": digest(path),
                          "videos": len(videos), "lesions": len(identities),
                          "images": len(data["images"]), "annotations": len(data["annotations"])}
        cohort.extend({"video_id": video, "split": split, "official_lesion_ids": lesions}
                      for video, lesions in sorted(videos.items()))
        del data
    return sorted(cohort, key=lambda item: item["video_id"]), sources


def sample_video(spec, config, video_info, output, required_starts=None):
    started = time.monotonic()
    video = spec["video_id"]
    archive_path = Path(config["annotation_directory"]) / f"{video}_annotations.tar.gz"
    expected_lesions = set(spec["official_lesion_ids"])
    expected_frames = int(video_info["num_frames"])
    fps = float(video_info["fps"])
    frames, visible = {}, defaultdict(list)
    counters = Counter()
    exclusions, crop_outside, filename_aliases = [], [], []
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            if not safe_member(member.name):
                raise ValueError(f"unsafe annotation member: {member.name}")
            if not member.isfile() or not member.name.endswith(".xml"):
                continue
            match = FRAME_RE.search(member.name)
            if match is None:
                raise ValueError(f"unrecognized annotation filename: {member.name}")
            index = int(match.group(1))
            if index in frames:
                raise ValueError(f"duplicate frame index: {video}/{index}")
            payload = archive.extractfile(member)
            if payload is None:
                raise ValueError(f"unreadable XML member: {member.name}")
            width, height, boxes, anomalies, raw_count = parse_xml(payload.read())
            if set(identity for identity, _ in boxes) - expected_lesions:
                raise ValueError(f"XML lesion absent from official partition: {video}/{index}")
            frame = {"frame_index": index, "xml_member": member.name, "width": width, "height": height,
                     "boxes_xyxy": [{"lesion_id": identity, "box": list(box)} for identity, box in boxes]}
            frame["annotation_eligible"] = not anomalies
            frame["visible_lesion_ids"] = []
            counters["xml_frames"] += 1
            counters["raw_boxes"] += raw_count
            if match.group(2):
                filename_aliases.append({"frame_index": index, "xml_member": member.name})
            if anomalies:
                exclusions.append({"frame_index": index, "anomalies": anomalies})
                counters["annotation_anomaly_frames"] += 1
            fallback_count = 0
            for record in frame["boxes_xyxy"]:
                x0, y0, x1, y1 = record["box"]
                counters["image_boundary_boxes"] += x0 == 0 or y0 == 0 or x1 == width or y1 == height
                counters["partly_crop_truncated_boxes"] += (x0 < 0.1 * width < x1) or (x0 < 0.9 * width < x1)
                mask, small_count = project_boxes({**frame, "boxes_xyxy": [record]})
                fallback_count += small_count
                if mask.any():
                    frame["visible_lesion_ids"].append(record["lesion_id"])
                    if not anomalies:
                        visible[record["lesion_id"]].append(index)
                else:
                    counters["fully_crop_outside_boxes"] += 1
                    crop_outside.append({"frame_index": index, **record})
            frame["small_box_fallback_count"] = fallback_count
            counters["small_box_fallback_boxes"] += fallback_count
            frame["class"] = 1 if frame["visible_lesion_ids"] else (0 if not boxes else -1)
            counters["all_boxes_outside_crop_frames"] += frame["class"] == -1
            counters["no_box_frames"] += not boxes
            frames[index] = frame
            if counters["xml_frames"] % 20000 == 0:
                print(f"XML {video}: {counters['xml_frames']}/{expected_frames}", flush=True)
    if not frames or len(frames) != expected_frames:
        raise ValueError(f"XML frame count mismatch: {video}: {len(frames)} != {expected_frames}")
    first, last = min(frames), max(frames)
    missing = sorted(set(range(first, last + 1)) - frames.keys())
    negative_indices = [index for index, frame in frames.items()
                        if frame["annotation_eligible"] and not frame["boxes_xyxy"]]
    selected = {}
    lesion_reports = []
    length = config["frames_per_clip"]

    def add_clip(start, label, identity=None):
        indices = list(range(start, start + length))
        clip_id = f"{video}_{start}"
        if clip_id not in selected:
            selected[clip_id] = {"clip_id": clip_id, "video_id": video, "split": spec["split"],
                                 "class": label, "fps": fps, "start_frame": start,
                                 "end_frame": indices[-1], "start_seconds": start / fps,
                                 "sampling_lesion_ids": [], "frames": [frames[i] for i in indices],
                                 "historical_exposure": "former_four_video_development" if video in config["former_development_videos"]
                                 else ("development_validation" if spec["split"] == "val" else "training"),
                                 "annotation_archive": str(archive_path)}
        clip = selected[clip_id]
        if clip["class"] != label:
            raise ValueError("positive and negative clip identity conflict")
        if identity is not None:
            if not all(identity in frames[i]["visible_lesion_ids"] for i in indices):
                raise ValueError("selected lesion lacks visible support in selected clip")
            clip["sampling_lesion_ids"].append(identity)

    for identity in sorted(expected_lesions):
        candidates, tail_count = candidate_blocks(visible[identity], length)
        required = sorted((required_starts or {}).get(identity, []))
        if not set(required) <= set(candidates) or len(required) > config["positive_clips_per_lesion"]:
            raise ValueError("Required observations are outside the eligible candidates or budget")
        additional = temporal_selection([start for start in candidates if start not in required],
                                        config["positive_clips_per_lesion"] - len(required))
        chosen = sorted(required + additional)
        for start in chosen:
            add_clip(start, 1, identity)
        lesion_reports.append({"lesion_id": identity, "visible_eligible_frames": len(visible[identity]),
                               "candidate_nonoverlapping_clips": len(candidates), "interval_tail_frames": tail_count,
                               "selected_starts": chosen, "selected_clips": len(chosen),
                               "requested_clips": config["positive_clips_per_lesion"],
                               "shortfall": max(0, config["positive_clips_per_lesion"] - len(chosen))})
    candidates, tail_count = candidate_blocks(negative_indices, length)
    chosen = temporal_selection(candidates, config["negative_clips_per_video"])
    for start in chosen:
        add_clip(start, 0)
    clips = sorted(selected.values(), key=lambda clip: clip["start_frame"])
    for clip in clips:
        if not all(frame["annotation_eligible"] for frame in clip["frames"]):
            raise ValueError("selected clip contains annotation anomaly")
        if clip["class"] == 0 and any(frame["boxes_xyxy"] for frame in clip["frames"]):
            raise ValueError("negative clip contains annotated box")
    positive_occurrences = sum(row["selected_clips"] for row in lesion_reports)
    summary = {**spec, "fps": fps, "counters": dict(counters), "first_xml_index": first,
               "last_xml_index": last, "missing_xml_indices": missing,
               "annotation_archive": str(archive_path), "archive_bytes": archive_path.stat().st_size,
               "archive_sha256": digest(archive_path),
               "lesions": lesion_reports, "positive_clips": sum(row["class"] == 1 for row in clips),
               "deduplicated_shared_positive_clips": positive_occurrences - sum(row["class"] == 1 for row in clips),
               "negative": {"candidate_nonoverlapping_clips": len(candidates), "interval_tail_frames": tail_count,
                            "selected_starts": chosen, "selected_clips": len(chosen),
                            "shortfall": max(0, config["negative_clips_per_video"] - len(chosen))},
               "seconds": time.monotonic() - started}
    write_json(output / f"{video}_exclusions.json", {"annotation_anomalies": exclusions,
               "fully_crop_outside_boxes": crop_outside, "integer_filename_aliases": filename_aliases,
               "missing_xml_indices": missing})
    write_json(output / f"{video}.json", {"summary": summary, "clips": clips})
    print(f"DONE {video}: {len(lesion_reports)} lesions, {summary['positive_clips']} positive + {len(chosen)} negative clips; {summary['seconds']:.1f}s", flush=True)
    return summary, clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="results/runs/20260927T0850Z_rc27_task_cohort_v1")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run = ROOT / args.run
    config = json.loads((run / "config.json").read_text())
    output = run / ("smoke" if args.smoke else "metadata")
    output.mkdir(parents=True, exist_ok=True)
    config_hash, script_hash = digest(run / "config.json"), digest(__file__)
    stamp_path = output / "input_identity.json"
    identity = {"config_sha256": config_hash, "script_sha256": script_hash}
    if stamp_path.exists() and json.loads(stamp_path.read_text()) != identity:
        raise ValueError("existing output uses different code or configuration")
    write_json(stamp_path, identity)
    cohort, sources = official_cohort(config)
    if args.smoke:
        cohort = [next(spec for spec in cohort if spec["split"] == split) for split in ("train", "val")]
    with (Path(config["annotation_directory"]) / "video_info.csv").open(encoding="utf-8-sig", newline="") as stream:
        video_info = {row["unique_video_name"]: row for row in csv.DictReader(stream)}
    summaries, clips = [], []
    started = time.monotonic()
    for position, spec in enumerate(cohort, 1):
        print(f"VIDEO {position}/{len(cohort)} {spec['video_id']} ({spec['split']})", flush=True)
        checkpoint = output / f"{spec['video_id']}.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            summary, sampled = saved["summary"], saved["clips"]
        else:
            summary, sampled = sample_video(spec, config, video_info[spec["video_id"]], output)
        summaries.append(summary)
        clips.extend(sampled)
    clips.sort(key=lambda clip: (clip["video_id"], clip["start_frame"]))
    if len({row["clip_id"] for row in clips}) != len(clips):
        raise ValueError("duplicate manifest clip")
    manifest = output / "clip_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in clips), encoding="utf-8")
    totals = {}
    for split in ("train", "val"):
        videos = [row for row in summaries if row["split"] == split]
        lesions = [lesion for row in videos for lesion in row["lesions"]]
        totals[split] = {"videos": len(videos), "official_lesions": len(lesions),
                         "lesions_with_selected_clips": sum(row["selected_clips"] > 0 for row in lesions),
                         "lesions_with_four_clips": sum(row["selected_clips"] == 4 for row in lesions),
                         "lesion_clip_shortfall": sum(row["shortfall"] for row in lesions),
                         "positive_clips": sum(row["positive_clips"] for row in videos),
                         "negative_clips": sum(row["negative"]["selected_clips"] for row in videos),
                         "negative_clip_shortfall": sum(row["negative"]["shortfall"] for row in videos)}
    result = {"status": "metadata_sampling_complete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "smoke": args.smoke, "sources": sources, "identity": identity,
              "totals": totals, "clips": len(clips), "frames": 8 * len(clips),
              "per_video": summaries, "manifest_sha256": digest(manifest),
              "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
                          "elapsed_seconds": time.monotonic() - started, "cpu_threads": 1,
                          "pixels_read": 0, "test_assets_read": 0}}
    write_json(output / "summary.json", result)
    print(json.dumps({"status": result["status"], "totals": totals, "clips": len(clips)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
