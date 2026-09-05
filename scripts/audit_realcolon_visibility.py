"""Audit REAL-Colon annotations for lesion-identity disappearance/reappearance events.

The audit streams the official compressed VOC annotations without extracting
millions of small XML files.  A missing box between two runs of the same
``unique_id`` is only a censoring *candidate*: image review is required to
separate occlusion, out-of-view, blur, annotation error, and other causes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tarfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


FRAME_RE = re.compile(r"_(\d+)(\.0)?\.xml$")


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not row.get(key) for row in rows):
        raise ValueError(f"missing rows or key {key!r} in {path}")
    result = {row[key]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate {key!r} in {path}")
    return result


def safe_member(name: str) -> bool:
    value = PurePosixPath(name)
    return not value.is_absolute() and ".." not in value.parts


def parse_xml(payload: bytes) -> tuple[int, int, list[tuple[str, tuple[int, int, int, int]]], list[dict], int]:
    root = ET.fromstring(payload)
    width = int(root.findtext("./size/width", "0"))
    height = int(root.findtext("./size/height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError("non-positive image dimensions")
    raw_boxes = []
    anomalies = []
    for node in root.findall("object"):
        unique_id = node.findtext("unique_id")
        box = node.find("bndbox")
        if not unique_id or box is None:
            raise ValueError("object missing unique_id or bndbox")
        coords = tuple(int(round(float(box.findtext(name, "nan")))) for name in ("xmin", "ymin", "xmax", "ymax"))
        left, top, right, bottom = coords
        clipped = (max(0, left), max(0, top), min(width, right), min(height, bottom))
        if clipped != coords:
            anomalies.append({
                "type": "box_clipped_to_frame_for_border_heuristic",
                "unique_id": unique_id,
                "original_box": list(coords),
                "clipped_box": list(clipped),
                "image_size": [width, height],
            })
        if not (0 <= clipped[0] < clipped[2] <= width and 0 <= clipped[1] < clipped[3] <= height):
            raise ValueError(f"invalid box after clipping {coords} -> {clipped} for {width}x{height}")
        raw_boxes.append((unique_id, clipped))

    grouped = defaultdict(list)
    for unique_id, box in raw_boxes:
        grouped[unique_id].append(box)
    boxes = []
    for unique_id, identity_boxes in grouped.items():
        if len(identity_boxes) > 1:
            anomalies.append({
                "type": "duplicate_identity_boxes_in_frame",
                "unique_id": unique_id,
                "box_count": len(identity_boxes),
                "boxes": [list(box) for box in identity_boxes],
                "resolution": "union_for_border_heuristic; binary identity presence retained once",
            })
        boxes.append((unique_id, (
            min(box[0] for box in identity_boxes), min(box[1] for box in identity_boxes),
            max(box[2] for box in identity_boxes), max(box[3] for box in identity_boxes),
        )))
    return width, height, boxes, anomalies, len(raw_boxes)

def contiguous_runs(observations: list[dict]) -> list[dict]:
    runs = []
    start = 0
    for index in range(1, len(observations) + 1):
        if index == len(observations) or observations[index]["frame"] != observations[index - 1]["frame"] + 1:
            block = observations[start:index]
            runs.append({
                "start_frame": block[0]["frame"],
                "end_frame": block[-1]["frame"],
                "length_frames": len(block),
                "start_box": block[0]["box"],
                "end_box": block[-1]["box"],
                "width": block[-1]["width"],
                "height": block[-1]["height"],
            })
            start = index
    return runs


def touches_border(box: list[int] | tuple[int, ...], width: int, height: int, margin: float) -> bool:
    left, top, right, bottom = box
    return min(left / width, top / height, (width - right) / width, (height - bottom) / height) <= margin


def audit_archive(path: Path, video: dict[str, str], lesions: set[str], min_anchor: int, border_margin: float) -> dict:
    video_id = video["unique_video_name"]
    expected_frames = int(video["num_frames"])
    fps = float(video["fps"])
    observations: dict[str, list[dict]] = defaultdict(list)
    xml_count = object_count = frames_with_boxes = multi_object_frames = 0
    bad_member_count = 0
    annotation_anomalies = []
    filename_anomalies = []

    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if not safe_member(member.name):
                bad_member_count += 1
                continue
            if not member.isfile() or not member.name.endswith(".xml"):
                continue
            match = FRAME_RE.search(member.name)
            if not match:
                raise ValueError(f"unparseable frame member: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"cannot read member: {member.name}")
            width, height, boxes, anomalies, raw_box_count = parse_xml(extracted.read())
            frame = int(match.group(1))
            if match.group(2):
                filename_anomalies.append({"frame": frame, "member": member.name, "type": "integer_frame_with_dot_zero_suffix"})
            xml_count += 1
            object_count += raw_box_count
            frames_with_boxes += bool(boxes)
            multi_object_frames += len(boxes) > 1
            for anomaly in anomalies:
                annotation_anomalies.append({"frame": frame, **anomaly})
            for unique_id, box in boxes:
                if unique_id not in lesions:
                    raise ValueError(f"annotation identity absent from lesion_info.csv: {unique_id}")
                observations[unique_id].append({"frame": frame, "box": list(box), "width": width, "height": height})

    if bad_member_count:
        raise ValueError(f"unsafe archive members in {path.name}: {bad_member_count}")
    if xml_count != expected_frames:
        raise ValueError(f"frame-count mismatch for {video_id}: {xml_count} != {expected_frames}")

    lesion_summaries = []
    gaps = []
    for lesion_id, items in sorted(observations.items()):
        ordered = sorted(items, key=lambda item: item["frame"])
        if len({item["frame"] for item in ordered}) != len(ordered):
            raise ValueError(f"duplicate frame for lesion {lesion_id}")
        runs = contiguous_runs(ordered)
        lesion_summaries.append({
            "lesion_id": lesion_id,
            "annotated_frames": len(ordered),
            "tracklet_count_frame_contiguity": len(runs),
            "first_frame": ordered[0]["frame"],
            "last_frame": ordered[-1]["frame"],
        })
        for index, (pre, post) in enumerate(zip(runs, runs[1:]), 1):
            gap_frames = post["start_frame"] - pre["end_frame"] - 1
            gaps.append({
                "gap_id": f"{lesion_id}:gap{index}",
                "video_id": video_id,
                "lesion_id": lesion_id,
                "pre_start_frame": pre["start_frame"],
                "pre_end_frame": pre["end_frame"],
                "pre_length_frames": pre["length_frames"],
                "gap_start_frame": pre["end_frame"] + 1,
                "gap_end_frame": post["start_frame"] - 1,
                "gap_frames": gap_frames,
                "gap_seconds": round(gap_frames / fps, 6),
                "post_start_frame": post["start_frame"],
                "post_end_frame": post["end_frame"],
                "post_length_frames": post["length_frames"],
                "anchors_at_border": {
                    "pre": touches_border(pre["end_box"], pre["width"], pre["height"], border_margin),
                    "post": touches_border(post["start_box"], post["width"], post["height"], border_margin),
                },
                "candidate_after_anchor_filter": pre["length_frames"] >= min_anchor and post["length_frames"] >= min_anchor,
                "censoring_mechanism": "unresolved_without_frames",
            })

    return {
        "video_id": video_id,
        "study_id": video_id.split("-")[0],
        "fps": fps,
        "frame_count": xml_count,
        "frames_with_boxes": frames_with_boxes,
        "object_count": object_count,
        "multi_object_frames": multi_object_frames,
        "metadata_num_lesions": int(video["num_lesions"]),
        "annotated_lesion_count": len(observations),
        "lesions": lesion_summaries,
        "gaps": gaps,
        "annotation_anomaly_count": len(annotation_anomalies),
        "annotation_anomaly_examples": annotation_anomalies[:25],
        "filename_anomaly_count": len(filename_anomalies),
        "filename_anomaly_examples": filename_anomalies[:25],
        "archive": {"name": path.name, "size_bytes": path.stat().st_size, "sha256": file_hash(path)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-anchor-frames", type=int, default=30)
    parser.add_argument("--border-margin", type=float, default=0.02)
    args = parser.parse_args()

    root = args.annotation_dir.resolve()
    video_path = root / "video_info.csv"
    lesion_path = root / "lesion_info.csv"
    videos = read_csv(video_path, "unique_video_name")
    lesion_rows = read_csv(lesion_path, "unique_object_id")
    lesions_by_video: dict[str, set[str]] = defaultdict(set)
    for lesion_id, row in lesion_rows.items():
        lesions_by_video[row["unique_video_name"]].add(lesion_id)
    archives = sorted(root.glob("*_annotations.tar.gz"))
    if len(archives) != len(videos) or len(videos) != 60:
        raise SystemExit(f"expected 60 archives/videos, found {len(archives)}/{len(videos)}")

    reports = []
    for index, archive in enumerate(archives, 1):
        video_id = archive.name.removesuffix("_annotations.tar.gz")
        if video_id not in videos:
            raise ValueError(f"archive has no video_info row: {archive.name}")
        report = audit_archive(
            archive, videos[video_id], lesions_by_video[video_id],
            args.minimum_anchor_frames, args.border_margin,
        )
        reports.append(report)
        print(f"[{index}/{len(archives)}] {index * 100 // len(archives)}% {video_id}", flush=True)

    all_gaps = [gap for report in reports for gap in report["gaps"]]
    candidate_gaps = [gap for gap in all_gaps if gap["candidate_after_anchor_filter"]]
    thresholds = [1, 5, 15, 30, 60, 150]
    report = {
        "schema_version": "endosae.realcolon-visibility-a0.v0",
        "status": "exploratory-complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "fallback_used": False,
        "source": {
            "figshare_article_id": 22202866,
            "figshare_doi": "10.25452/figshare.plus.22202866",
            "annotation_dir": str(root),
            "video_info_sha256": file_hash(video_path),
            "lesion_info_sha256": file_hash(lesion_path),
        },
        "event_rule": {
            "identity": "official unique_object_id in VOC object annotations",
            "tracklet": "maximal run of consecutive frames with a bbox for the same identity",
            "gap": "one or more frames without that identity between two tracklets of the same identity",
            "minimum_anchor_frames": args.minimum_anchor_frames,
            "border_margin_fraction": args.border_margin,
            "important_boundary": "gap proves annotation absence and later re-identification, not the visual cause of absence",
        },
        "aggregate": {
            "video_count": len(reports),
            "study_count": len({report["study_id"] for report in reports}),
            "metadata_lesion_count": len(lesion_rows),
            "frame_count": sum(report["frame_count"] for report in reports),
            "annotated_frame_count": sum(report["frames_with_boxes"] for report in reports),
            "box_count": sum(report["object_count"] for report in reports),
            "videos_with_zero_lesions": sum(report["metadata_num_lesions"] == 0 for report in reports),
            "lesions_with_multiple_tracklets": sum(lesion["tracklet_count_frame_contiguity"] > 1 for report in reports for lesion in report["lesions"]),
            "all_identity_gaps": len(all_gaps),
            "anchor_filtered_gaps": len(candidate_gaps),
            "anchor_filtered_gap_counts_by_minimum_length": {
                str(threshold): sum(gap["gap_frames"] >= threshold for gap in candidate_gaps)
                for threshold in thresholds
            },
            "anchor_filtered_gaps_with_both_border_anchors": sum(
                gap["anchors_at_border"]["pre"] and gap["anchors_at_border"]["post"] for gap in candidate_gaps
            ),
            "archive_bytes": sum(report["archive"]["size_bytes"] for report in reports),
            "annotation_anomaly_count": sum(report["annotation_anomaly_count"] for report in reports),
            "filename_anomaly_count": sum(report["filename_anomaly_count"] for report in reports),
        },
        "claim_boundary": {
            "allowed": [
                "same-lesion disappearance/reappearance candidate generation",
                "patient/video-grouped external validation design",
                "multi-study and device-stratified evaluation design",
            ],
            "not_allowed_without_frame_review": [
                "occlusion versus out-of-view versus blur mechanism labels",
                "true absence during a bbox-free interval",
                "visibility-aware SAE result",
                "causal or mechanistic EndoFM claim",
            ],
        },
        "candidate_gaps": candidate_gaps,
        "videos": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print("[60/60] 100% complete", flush=True)
    print(json.dumps(report["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
