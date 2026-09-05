"""Build a blinded REAL-Colon visibility-candidate discovery pack.

The builder selectively extracts frames from one official tar.gz archive,
preserves source bytes, separates blind items from the private identity map,
and never converts a missing box into an occlusion label.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import random
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, BinaryIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCHEMA = "endosae.realcolon-visibility-pack.v0"
SEED = "endosae-realcolon-discovery-v0"
REVIEW_SCHEMA = "endosae.realcolon-visibility-review.v1"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ProgressFile:
    def __init__(self, path: Path, log_path: Path, label: str = "archive") -> None:
        self.path = path
        self.raw: BinaryIO = path.open("rb")
        self.total = path.stat().st_size
        self.log_path = log_path
        self.label = label
        self.last_percent = -1
        self.started = time.perf_counter()
        self._emit(0, "initialized compressed-byte scan")

    def _emit(self, percent: int, message: str) -> None:
        line = (
            f"PROGRESS {percent}% | {self.label} | {message} "
            f"({self.raw.tell()}/{self.total} compressed bytes)"
        )
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def read(self, size: int = -1) -> bytes:
        payload = self.raw.read(size)
        percent = min(100, int(100 * self.raw.tell() / self.total))
        if percent >= self.last_percent + 1 or (not payload and percent == 100):
            self.last_percent = percent
            self._emit(percent, "scanning archive and selecting requested frames")
        return payload

    def close(self) -> None:
        self.raw.close()


def _blind_id(gap_id: str) -> str:
    digest = hashlib.sha256(f"{SEED}|{gap_id}".encode("utf-8")).hexdigest()[:8]
    return f"A0-{digest.upper()}"


def _candidate_rows(summary: dict[str, Any], video_id: str, minimum_gap: int) -> list[dict[str, Any]]:
    rows = [
        row
        for row in summary["candidate_gaps"]
        if row["video_id"] == video_id
        and row["gap_frames"] >= minimum_gap
        and not row["anchors_at_border"]["pre"]
        and not row["anchors_at_border"]["post"]
    ]
    return sorted(rows, key=lambda row: _blind_id(row["gap_id"]))


def _load_multi_pack_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "endosae.realcolon-visibility-pilot.v1":
        raise RuntimeError("unsupported multi-pack config schema")
    summary_path = Path(payload["summary_path"]).resolve()
    acquisition_path = Path(payload["acquisition_metrics_path"]).resolve()
    if _sha256(summary_path) != payload["summary_sha256"]:
        raise RuntimeError("visibility A0 summary hash mismatch")
    if _sha256(acquisition_path) != payload["acquisition_metrics_sha256"]:
        raise RuntimeError("acquisition metrics hash mismatch")
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    acquired = {row["video_id"]: row for row in acquisition["completed_files"]}
    for spec in payload["archives"]:
        witness = acquired.get(spec["video_id"])
        if witness is None:
            raise RuntimeError(f"missing acquisition witness for {spec['video_id']}")
        for key in ("bytes", "md5", "sha256"):
            config_key = "expected_bytes" if key == "bytes" else key
            if spec[config_key] != witness[key]:
                raise RuntimeError(
                    f"acquisition witness mismatch for {spec['video_id']} field {key}"
                )
        if Path(spec["archive_path"]).resolve() != Path(witness["path"]).resolve():
            raise RuntimeError(f"archive path mismatch for {spec['video_id']}")
        if "verified" not in witness["status"]:
            raise RuntimeError(f"archive is not verified for {spec['video_id']}")
    return payload


def run(
    archive_specs: list[dict[str, Any]],
    summary_path: Path,
    output_dir: Path,
    run_dir: Path,
    minimum_gap: int,
    anchor_frames: int,
    source_config_path: Path | None = None,
) -> None:
    if output_dir.exists() or run_dir.exists():
        raise RuntimeError("refusing to overwrite output or run directory")
    output_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not archive_specs:
        raise RuntimeError("at least one archive specification is required")
    video_ids = [str(spec["video_id"]) for spec in archive_specs]
    if len(video_ids) != len(set(video_ids)):
        raise RuntimeError("duplicate video_id in archive specifications")
    normalized_specs = []
    for spec in archive_specs:
        archive_path = Path(spec["archive_path"]).resolve()
        if not archive_path.is_file():
            raise RuntimeError(f"archive not found: {archive_path}")
        expected_bytes = spec.get("expected_bytes")
        if expected_bytes is not None and archive_path.stat().st_size != int(expected_bytes):
            raise RuntimeError(
                f"archive size mismatch for {spec['video_id']}: "
                f"{archive_path.stat().st_size} != {expected_bytes}"
            )
        normalized_specs.append({**spec, "archive_path": archive_path})
    candidates = []
    for video_id in video_ids:
        candidates.extend(_candidate_rows(summary, video_id, minimum_gap))
    candidates.sort(key=lambda row: _blind_id(row["gap_id"]))
    if not candidates:
        raise RuntimeError("no candidates satisfy the frozen rule")

    requested: dict[str, dict[str, Any]] = {}
    blind_items = []
    private_map = []
    for row in candidates:
        blind_id = _blind_id(row["gap_id"])
        video_id = row["video_id"]
        segments = {
            "pre": list(range(row["pre_end_frame"] - anchor_frames + 1, row["pre_end_frame"] + 1)),
            "middle": list(range(row["gap_start_frame"], row["gap_end_frame"] + 1)),
            "post": list(range(row["post_start_frame"], row["post_start_frame"] + anchor_frames)),
        }
        for role, frames in segments.items():
            for frame in frames:
                member = f"{video_id}_frames/{video_id}_{frame}.jpg"
                if member in requested:
                    raise RuntimeError(
                        f"overlapping candidate windows are not supported without explicit "
                        f"multi-item copying: {member}"
                    )
                requested[member] = {"blind_id": blind_id, "role": role, "frame": frame}
        blind_items.append(
            {
                "blind_item_id": blind_id,
                "segments": {role: {"frame_count": len(frames)} for role, frames in segments.items()},
                "source_frame_numbers_hidden_from_default_view": True,
                "candidate_only": True,
            }
        )
        private_map.append(
            {
                "blind_item_id": blind_id,
                "video_id": video_id,
                "gap_id": row["gap_id"],
                "lesion_id": row["lesion_id"],
                "segments": segments,
                "source_identity_evidence": "official REAL-Colon unique_id",
                "middle_label": "unresolved_without_human_review",
            }
        )

    if source_config_path:
        source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
        expected_candidates = source_config.get("expected_candidate_count")
        expected_frames = source_config.get("expected_requested_frame_count")
        if expected_candidates is not None and len(candidates) != int(expected_candidates):
            raise RuntimeError(
                f"candidate count mismatch: {len(candidates)} != {expected_candidates}"
            )
        if expected_frames is not None and len(requested) != int(expected_frames):
            raise RuntimeError(
                f"requested frame count mismatch: {len(requested)} != {expected_frames}"
            )

    found: dict[str, dict[str, Any]] = {}
    archive_records = []
    for spec in normalized_specs:
        archive = spec["archive_path"]
        video_id = str(spec["video_id"])
        progress = ProgressFile(archive, run_dir / "stdout.log", label=video_id)
        found_before = len(found)
        try:
            with tarfile.open(fileobj=progress, mode="r|gz") as stream:
                for member in stream:
                    meta = requested.get(member.name)
                    if meta is None:
                        continue
                    if not member.isfile() or member.size <= 0:
                        raise RuntimeError(f"invalid requested archive member: {member.name}")
                    source = stream.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"cannot read requested member: {member.name}")
                    target_dir = output_dir / "frames" / meta["blind_id"] / meta["role"]
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target = target_dir / f"{meta['frame']:06d}.jpg"
                    with target.open("wb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                    found[member.name] = {
                        **meta,
                        "relative_path": target.relative_to(output_dir).as_posix(),
                        "bytes": target.stat().st_size,
                        "sha256": _sha256(target),
                    }
        finally:
            progress.close()
        archive_records.append({
            "video_id": video_id,
            "archive_path": str(archive),
            "archive_bytes": archive.stat().st_size,
            "archive_md5_from_acquisition": spec.get("md5"),
            "archive_sha256_from_acquisition": spec.get("sha256"),
            "extracted_frame_count": len(found) - found_before,
        })

    missing = sorted(set(requested) - set(found))
    if missing:
        raise RuntimeError(f"archive is missing {len(missing)} requested frames; first={missing[0]}")

    from PIL import Image

    decode_failures = []
    dimensions: dict[str, int] = {}
    for record in found.values():
        path = output_dir / record["relative_path"]
        try:
            with Image.open(path) as image:
                image.load()
                key = f"{image.mode}:{image.width}x{image.height}"
                dimensions[key] = dimensions.get(key, 0) + 1
        except Exception as exc:  # pragma: no cover - real asset failure path
            decode_failures.append({"path": record["relative_path"], "error": str(exc)})
    if decode_failures:
        raise RuntimeError(f"JPEG decode failures: {len(decode_failures)}")

    _write_json(output_dir / "blind_manifest.json", {
        "schema_version": SCHEMA,
        "video_id_hidden_in_default_view": True,
        "candidate_rule": f"gap>={minimum_gap}, nonborder anchors, {anchor_frames} pre/post frames",
        "items": blind_items,
    })
    _write_json(output_dir / "private_mapping.json", {
        "schema_version": SCHEMA,
        "warning": "keep separate from blinded annotation view",
        "items": private_map,
    })
    _write_json(run_dir / "metrics.json", {
        "schema_version": SCHEMA,
        "status": "discovery-pack-complete",
        "video_count": len(video_ids),
        "candidate_count": len(candidates),
        "requested_frame_count": len(requested),
        "extracted_frame_count": len(found),
        "decode_failure_count": 0,
        "dimension_counts": dimensions,
        "archive_records": archive_records,
        "output_dir": str(output_dir),
        "blind_manifest_sha256": _sha256(output_dir / "blind_manifest.json"),
        "private_mapping_sha256": _sha256(output_dir / "private_mapping.json"),
        "fallback": False,
        "limitations": [
            "four-study candidate-yield pilot; not prevalence or clinical evidence",
            "candidate generation uses annotation gaps but not censoring labels",
            "human review is required before any visible/censored episode claim",
        ],
    })
    _write_json(run_dir / "config.json", {
        "schema_version": SCHEMA,
        "video_ids": video_ids,
        "archive_specs": [
            {
                **{key: value for key, value in spec.items() if key != "archive_path"},
                "archive_path": str(spec["archive_path"]),
            }
            for spec in normalized_specs
        ],
        "source_config_path": str(source_config_path) if source_config_path else None,
        "source_config_sha256": _sha256(source_config_path) if source_config_path else None,
        "minimum_gap_frames": minimum_gap,
        "anchor_frames": anchor_frames,
        "blind_seed": SEED,
        "fallback_allowed": False,
    })
    _write_json(run_dir / "status.json", {
        "schema_version": "endosae.run-status.v0",
        "run_id": run_dir.name,
        "status": "discovery-pack-complete",
        "fallback": False,
        "assets": ["config.json", "metrics.json", "stdout.log"],
    })


def _review_html(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EndoSAE blinded visibility calibration</title>
<style>
:root{color-scheme:dark;--bg:#11151b;--panel:#1b222c;--text:#eef3f8;--muted:#aeb9c6;--accent:#55c2ff;--warn:#ffcc66}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.45 system-ui,sans-serif}
main{max-width:1280px;margin:auto;padding:18px}.bar,.grid{display:grid;gap:12px}.bar{grid-template-columns:auto 1fr auto}.grid{grid-template-columns:minmax(0,2fr) minmax(320px,1fr)}
.panel{background:var(--panel);border:1px solid #34404e;border-radius:10px;padding:14px}button,select,input,textarea{font:inherit}
button{background:#273444;color:var(--text);border:1px solid #48596c;border-radius:6px;padding:7px 11px;cursor:pointer}
button:hover{border-color:var(--accent)}.viewer{position:relative;width:100%;background:#000;border-radius:8px;overflow:hidden}
#frame{display:block;width:100%;height:auto}#overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
#overlay rect{fill:none;stroke:#00e5ff;stroke-width:5;vector-effect:non-scaling-stroke}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}#scrub{flex:1;min-width:240px}
.notice{color:var(--warn)}label{display:block;margin:9px 0 3px;color:var(--muted)}fieldset{border:1px solid #44505d;margin:10px 0}
.check label{display:inline-block;margin-right:10px;color:var(--text)}textarea{width:100%;min-height:70px}
.status{white-space:pre-wrap;color:var(--muted)}@media(max-width:900px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body><main>
<div class="bar panel"><button id="prevItem">Previous item</button><strong id="itemTitle"></strong><button id="nextItem">Next item</button></div>
<p class="notice">Calibration only. Missing middle boxes are not absence or censoring labels. Anchor boxes are localization aids and are hidden by default.</p>
<div class="grid">
<section class="panel">
  <div class="viewer"><img id="frame" alt="blinded endoscopy frame"><svg id="overlay" viewBox="0 0 1352 1080"></svg></div>
  <div class="controls"><button id="prevFrame">◀</button><button id="play">Play</button><button id="nextFrame">▶</button><input id="scrub" type="range" min="0" value="0"><span id="frameLabel"></span></div>
  <div class="controls"><label><input id="showAnchor" type="checkbox"> Show official pre/post anchor localization</label></div>
  <p id="segmentInfo" class="status"></p>
</section>
<section class="panel">
  <label>Annotator ID<input id="annotator" value="calibration-01"></label>
  <label>Same target pre/post<select id="identity"><option>uncertain</option><option>yes</option><option>no</option></select></label>
  <label>Middle observability<select id="observability"><option>unknown</option><option>visible_full</option><option>visible_partial</option><option>censored_in_view</option><option>out_of_view</option></select></label>
  <fieldset class="check"><legend>Censoring mechanism (only for censored/out_of_view)</legend>
    <label><input type="checkbox" name="mechanism" value="instrument_occlusion">instrument</label>
    <label><input type="checkbox" name="mechanism" value="fluid_or_bubble">fluid/bubble</label>
    <label><input type="checkbox" name="mechanism" value="debris">debris</label>
    <label><input type="checkbox" name="mechanism" value="specularity_or_saturation">specularity</label>
    <label><input type="checkbox" name="mechanism" value="blur_or_fast_motion">blur/motion</label>
    <label><input type="checkbox" name="mechanism" value="lens_contamination">lens contamination</label>
    <label><input type="checkbox" name="mechanism" value="frame_boundary">frame boundary</label>
    <label><input type="checkbox" name="mechanism" value="other">other</label>
    <label><input type="checkbox" name="mechanism" value="unknown">unknown</label>
  </fieldset>
  <label>Middle-relative boundary start<input id="start" type="number" min="0"></label>
  <label>Middle-relative boundary end<input id="end" type="number" min="0"></label>
  <label>Confidence<select id="confidence"><option>1</option><option>2</option><option>3</option><option>4</option></select></label>
  <label>Exclude reason<select id="exclude"><option>identity_unresolved</option><option>none</option><option>no_true_censoring</option><option>insufficient_context</option><option>corrupt_or_missing_frames</option><option>other</option></select></label>
  <label>Notes<textarea id="notes"></textarea></label>
  <button id="saveItem">Save current item</button> <button id="export">Export JSONL</button>
  <p id="formStatus" class="status"></p>
</section></div>
</main><script>
const manifest=__MANIFEST__;let itemIndex=0,frameIndex=0,timer=null;const saved={};
const $=id=>document.getElementById(id);
function item(){return manifest.items[itemIndex]} function frames(){return item().frames}
function render(){
 const it=item(),fr=frames()[frameIndex];$("itemTitle").textContent=`${itemIndex+1}/${manifest.items.length} · ${it.blind_item_id}`;
 $("frame").src=fr.path;$("scrub").max=frames().length-1;$("scrub").value=frameIndex;
 $("frameLabel").textContent=`${fr.role} ${fr.role_rank+1}/${it.segment_counts[fr.role]}`;
 $("segmentInfo").textContent=`Chronological frame ${frameIndex+1}/${frames().length}. Boundaries entered below are zero-based indices within the middle segment.`;
 const svg=$("overlay");svg.innerHTML="";
 if($("showAnchor").checked&&fr.anchor_boxes) for(const b of fr.anchor_boxes){const r=document.createElementNS("http://www.w3.org/2000/svg","rect");r.setAttribute("x",b[0]);r.setAttribute("y",b[1]);r.setAttribute("width",b[2]-b[0]);r.setAttribute("height",b[3]-b[1]);svg.appendChild(r)}
}
function restore(){const r=saved[item().blind_item_id];if(!r){$("identity").value="uncertain";$("observability").value="unknown";$("exclude").value="identity_unresolved";$("confidence").value="1";$("start").value="";$("end").value="";$("notes").value="";document.querySelectorAll("[name=mechanism]").forEach(x=>x.checked=false);return}
 for(const k of ["identity","observability","exclude","confidence","start","end","notes"]) $(k).value=r[k]??"";document.querySelectorAll("[name=mechanism]").forEach(x=>x.checked=r.mechanisms.includes(x.value))}
function record(){const obs=$("observability").value,censored=["censored_in_view","out_of_view"].includes(obs);return{schema_version:"0.1.0",blind_item_id:item().blind_item_id,annotator_id:$("annotator").value.trim(),round_id:"calibration",same_target_pre_post:$("identity").value,middle_observability:obs,censoring_mechanisms:censored?[...document.querySelectorAll("[name=mechanism]:checked")].map(x=>x.value):[],boundary_start_frame:censored&&$("start").value!==""?Number($("start").value):null,boundary_end_frame:censored&&$("end").value!==""?Number($("end").value):null,confidence:Number($("confidence").value),exclude_reason:$("exclude").value,notes:$("notes").value.trim()||null}}
function save(){const r=record();saved[r.blind_item_id]={...r,identity:r.same_target_pre_post,observability:r.middle_observability,exclude:r.exclude_reason,start:r.boundary_start_frame,end:r.boundary_end_frame,mechanisms:r.censoring_mechanisms};$("formStatus").textContent="Saved locally in this page session. Export and run the Python validator before use."}
function moveItem(d){save();itemIndex=(itemIndex+d+manifest.items.length)%manifest.items.length;frameIndex=0;restore();render()}
$("prevItem").onclick=()=>moveItem(-1);$("nextItem").onclick=()=>moveItem(1);$("prevFrame").onclick=()=>{frameIndex=Math.max(0,frameIndex-1);render()};$("nextFrame").onclick=()=>{frameIndex=Math.min(frames().length-1,frameIndex+1);render()};
$("scrub").oninput=e=>{frameIndex=Number(e.target.value);render()};$("showAnchor").onchange=render;$("saveItem").onclick=save;
$("play").onclick=()=>{if(timer){clearInterval(timer);timer=null;$("play").textContent="Play";return}$("play").textContent="Pause";timer=setInterval(()=>{if(frameIndex>=frames().length-1){clearInterval(timer);timer=null;$("play").textContent="Play"}else{frameIndex++;render()}},100)};
$("export").onclick=()=>{save();const rows=manifest.items.map(x=>saved[x.blind_item_id]).filter(Boolean).map(x=>{const y={...x};for(const k of ["identity","observability","exclude","start","end","mechanisms"])delete y[k];return JSON.stringify(y)}).join("\n")+"\n";const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([rows],{type:"application/jsonl"}));a.download="visibility_annotations_calibration.jsonl";a.click();URL.revokeObjectURL(a.href)};
if(new URLSearchParams(location.search).get("anchors")==="1")$("showAnchor").checked=true;
restore();render();
</script></body></html>""".replace("__MANIFEST__", payload)


def build_review(
    pack_dir: Path,
    annotation_archive: Path,
    review_dir: Path,
    run_dir: Path,
) -> None:
    if review_dir.exists() or run_dir.exists():
        raise RuntimeError("refusing to overwrite review or run directory")
    review_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    private = json.loads((pack_dir / "private_mapping.json").read_text(encoding="utf-8"))
    blind = json.loads((pack_dir / "blind_manifest.json").read_text(encoding="utf-8"))
    blind_ids = {item["blind_item_id"] for item in blind["items"]}
    items = list(private["items"])
    if {item["blind_item_id"] for item in items} != blind_ids:
        raise RuntimeError("blind/private item mismatch")
    random.Random(SEED + "|review-v1").shuffle(items)

    if __package__:
        from .audit_realcolon_visibility import parse_xml
    else:
        from audit_realcolon_visibility import parse_xml

    review_items: list[dict[str, Any]] = []
    anchor_frame_count = 0
    anchor_box_count = 0
    duplicate_anchor_frame_count = 0
    annotation_archive_records = []
    with ExitStack() as stack:
        archives = {}
        for video_id in sorted({item["video_id"] for item in items}):
            archive_path = (
                annotation_archive
                if annotation_archive.is_file()
                else annotation_archive / f"{video_id}_annotations.tar.gz"
            )
            if not archive_path.is_file():
                raise RuntimeError(f"annotation archive not found: {archive_path}")
            archives[video_id] = stack.enter_context(tarfile.open(archive_path, mode="r:gz"))
            annotation_archive_records.append({
                "video_id": video_id,
                "path": str(archive_path),
                "bytes": archive_path.stat().st_size,
                "sha256": _sha256(archive_path),
            })
        for item in items:
            archive = archives[item["video_id"]]
            frames: list[dict[str, Any]] = []
            for role in ("pre", "middle", "post"):
                for rank, frame in enumerate(item["segments"][role]):
                    source = pack_dir / "frames" / item["blind_item_id"] / role / f"{frame:06d}.jpg"
                    if not source.is_file():
                        raise RuntimeError(f"missing extracted frame: {source}")
                    boxes: list[list[int]] = []
                    if role != "middle":
                        member_name = f"{item['video_id']}_annotations/{item['video_id']}_{frame}.xml"
                        member = archive.getmember(member_name)
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise RuntimeError(f"cannot read annotation member: {member_name}")
                        _, _, parsed, _, _ = parse_xml(extracted.read())
                        boxes = [list(box) for identity, box in parsed if identity == item["lesion_id"]]
                        if not boxes:
                            raise RuntimeError(f"target anchor missing: {item['blind_item_id']} {role} {rank}")
                        anchor_frame_count += 1
                        anchor_box_count += len(boxes)
                        duplicate_anchor_frame_count += int(len(boxes) > 1)
                    frames.append({
                        "role": role,
                        "role_rank": rank,
                        "path": f"../frames/{item['blind_item_id']}/{role}/{frame:06d}.jpg",
                        "anchor_boxes": boxes,
                    })
            review_items.append({
                "blind_item_id": item["blind_item_id"],
                "segment_counts": {role: len(item["segments"][role]) for role in ("pre", "middle", "post")},
                "frames": frames,
            })

    manifest = {
        "schema_version": REVIEW_SCHEMA,
        "round_id": "calibration",
        "randomization_seed_sha256": hashlib.sha256((SEED + "|review-v1").encode()).hexdigest(),
        "source_frame_numbers_hidden_from_ui": True,
        "middle_has_automatic_boxes_or_labels": False,
        "anchor_boxes_default_visible": False,
        "boundary_index_semantics": "zero-based index within middle segment",
        "items": review_items,
    }
    _write_json(review_dir / "review_manifest.json", manifest)
    (review_dir / "index.html").write_text(_review_html(manifest), encoding="utf-8")
    _write_json(run_dir / "config.json", {
        "schema_version": REVIEW_SCHEMA,
        "pack_dir": str(pack_dir),
        "annotation_archive": str(annotation_archive),
        "review_dir": str(review_dir),
        "fallback_allowed": False,
    })
    _write_json(run_dir / "metrics.json", {
        "schema_version": REVIEW_SCHEMA,
        "status": "calibration-review-pack-complete",
        "blind_item_count": len(review_items),
        "frame_count": sum(len(item["frames"]) for item in review_items),
        "anchor_frame_count": anchor_frame_count,
        "anchor_box_count": anchor_box_count,
        "duplicate_anchor_frame_count": duplicate_anchor_frame_count,
        "middle_automatic_box_count": 0,
        "annotation_archive_records": annotation_archive_records,
        "review_manifest_sha256": _sha256(review_dir / "review_manifest.json"),
        "viewer_sha256": _sha256(review_dir / "index.html"),
        "fallback": False,
    })
    _write_json(run_dir / "status.json", {
        "schema_version": "endosae.run-status.v0",
        "run_id": run_dir.name,
        "status": "calibration-review-pack-complete",
        "fallback": False,
        "assets": ["config.json", "metrics.json"],
    })


def validate_review_annotations(
    annotations_path: Path,
    review_manifest_path: Path,
    run_dir: Path,
) -> None:
    if run_dir.exists():
        raise RuntimeError("refusing to overwrite validation run directory")
    run_dir.mkdir(parents=True)
    from src.visibility_annotations import validate_annotation_record

    manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    expected = {item["blind_item_id"]: item for item in manifest["items"]}
    records = []
    for line_number, line in enumerate(annotations_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on line {line_number}: {exc}") from exc
        validate_annotation_record(record)
        records.append(record)
    observed_ids = [record["blind_item_id"] for record in records]
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("duplicate blind_item_id in annotation export")
    if set(observed_ids) != set(expected):
        raise ValueError(
            f"annotation item mismatch: missing={sorted(set(expected)-set(observed_ids))}, "
            f"unexpected={sorted(set(observed_ids)-set(expected))}"
        )
    for record in records:
        if record["middle_observability"] in {"censored_in_view", "out_of_view"}:
            middle_count = expected[record["blind_item_id"]]["segment_counts"]["middle"]
            if record["boundary_end_frame"] >= middle_count:
                raise ValueError(
                    f"boundary exceeds middle segment for {record['blind_item_id']}: "
                    f"{record['boundary_end_frame']} >= {middle_count}"
                )
    _write_json(run_dir / "metrics.json", {
        "schema_version": REVIEW_SCHEMA,
        "status": "annotation-validation-pass",
        "record_count": len(records),
        "expected_record_count": len(expected),
        "annotations_sha256": _sha256(annotations_path),
        "review_manifest_sha256": _sha256(review_manifest_path),
        "fallback": False,
    })
    _write_json(run_dir / "status.json", {
        "schema_version": "endosae.run-status.v0",
        "run_id": run_dir.name,
        "status": "annotation-validation-pass",
        "fallback": False,
        "assets": ["metrics.json"],
    })


def extract_task_manifest(manifest_path: Path, archive_dir: Path, output_dir: Path, run_dir: Path,
                          reuse_run: Path | None = None, available_only: bool = False) -> None:
    """Consume an already selected task manifest without selecting new examples.

    Reuses this module's sequential archive reader and byte-preserving extraction.
    Per-video completion receipts permit continuation after a long archive scan.
    """
    from PIL import Image
    import re

    manifest_hash = _sha256(manifest_path)
    clips = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    requested = {}
    for clip in clips:
        video = clip["video_id"]
        if re.fullmatch(r"\d{3}-\d{3}", video) is None:
            raise ValueError("invalid video id")
        for frame in clip["frames"]:
            index = frame["frame_index"]
            if type(index) is not int or index < 0:
                raise ValueError("invalid frame index")
            key = f"{video}_frames/{video}_{index}.jpg"
            if key in requested:
                raise ValueError("task manifest contains overlapping frames")
            requested[key] = (video, frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    receipts = []
    pending = []
    reuse_clips = []
    reuse_summary = None
    if reuse_run is not None:
        reuse_summary = json.loads((reuse_run / "extraction_summary.json").read_text(encoding="utf-8"))
        reuse_manifest = reuse_run / "clip_manifest.jsonl"
        if _sha256(reuse_manifest) != reuse_summary["manifest_sha256"]:
            raise RuntimeError("source extraction manifest changed")
        reuse_clips = [json.loads(line) for line in reuse_manifest.read_text(encoding="utf-8").splitlines()]
    for video in sorted({value[0] for value in requested.values()}):
        receipt_path = run_dir / f"extracted_{video}.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["manifest_sha256"] != manifest_hash:
                raise RuntimeError("completed extraction belongs to a different manifest")
            for frame in receipt["frames"]:
                path = output_dir / frame["relative_path"]
                if not path.is_file() or path.stat().st_size != frame["bytes"]:
                    raise RuntimeError("completed extracted asset missing or changed")
            receipts.append(receipt)
            continue
        wanted = {key: value[1] for key, value in requested.items() if value[0] == video}
        reuse_receipt_path = reuse_run / f"extracted_{video}.json" if reuse_run else None
        if reuse_receipt_path is not None and reuse_receipt_path.exists():
            old_clips = [row for row in reuse_clips if row["video_id"] == video]
            new_clips = [row for row in clips if row["video_id"] == video]
            if old_clips != new_clips:
                raise RuntimeError("reused video clips or labels differ from source run")
            old = json.loads(reuse_receipt_path.read_text(encoding="utf-8"))
            if old["manifest_sha256"] != reuse_summary["manifest_sha256"]:
                raise RuntimeError("reused receipt belongs to a different source manifest")
            if {frame["archive_member"] for frame in old["frames"]} != set(wanted) or len(old["frames"]) != len(wanted):
                raise RuntimeError("reused receipt frame coverage mismatch")
            for frame in old["frames"]:
                relative = Path(video) / f"{frame['frame_index']:06d}.jpg"
                if frame["relative_path"] != relative.as_posix() or frame["archive_member"] != f"{video}_frames/{video}_{frame['frame_index']}.jpg":
                    raise RuntimeError("invalid reused frame path")
                source = Path(reuse_summary["output_dir"]) / relative
                destination = output_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_suffix(".jpg.partial")
                shutil.copyfile(source, partial)
                if partial.stat().st_size != frame["bytes"] or _sha256(partial) != frame["sha256"]:
                    raise RuntimeError("reused source bytes changed")
                partial.replace(destination)
            receipt = dict(old, manifest_sha256=manifest_hash,
                           reused_from={"run": str(reuse_run), "manifest_sha256": old["manifest_sha256"],
                                        "receipt_sha256": _sha256(reuse_receipt_path)})
            _write_json(receipt_path, receipt)
            receipts.append(receipt)
            print(f"REUSED {video}: {len(old['frames'])} verified frames", flush=True)
            continue
        remaining = set(wanted)
        found = []
        archive = archive_dir / f"{video}_frames.tar.gz"
        if available_only and not archive.is_file():
            pending.append(video)
            continue
        reader = ProgressFile(archive, run_dir / "extraction_stdout.log", label=video)
        try:
            with tarfile.open(fileobj=reader, mode="r|gz") as stream:
                for member in stream:
                    if member.name not in wanted:
                        continue
                    if member.name not in remaining:
                        raise RuntimeError("duplicate requested archive member")
                    if not member.isfile() or member.size <= 0:
                        raise RuntimeError("invalid requested frame")
                    meta = wanted[member.name]
                    relative = Path(video) / f"{meta['frame_index']:06d}.jpg"
                    destination = output_dir / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    partial = destination.with_suffix(".jpg.partial")
                    with stream.extractfile(member) as source, partial.open("wb") as target:
                        shutil.copyfileobj(source, target, length=1024 * 1024)
                    with Image.open(partial) as image:
                        image.load()
                        if image.size != (meta["width"], meta["height"]):
                            raise RuntimeError("image dimensions disagree with corresponding XML")
                    partial.replace(destination)
                    found.append({"frame_index": meta["frame_index"], "archive_member": member.name,
                                  "relative_path": relative.as_posix(), "bytes": destination.stat().st_size,
                                  "sha256": _sha256(destination)})
                    remaining.remove(member.name)
                    if not remaining:
                        break
        finally:
            reader.close()
        if remaining:
            raise RuntimeError(f"missing requested frames: {sorted(remaining)[:3]}")
        receipt = {"video_id": video, "manifest_sha256": manifest_hash, "archive_bytes": archive.stat().st_size,
                   "frames": sorted(found, key=lambda frame: frame["frame_index"])}
        _write_json(receipt_path, receipt)
        receipts.append(receipt)
    _write_json(run_dir / "extraction_summary.json", {
        "status": "PARTIAL_WAITING_FOR_ARCHIVES" if pending else "EXTRACTED_DIMENSIONS_VERIFIED", "manifest_sha256": manifest_hash,
        "output_dir": str(output_dir), "frames": sum(len(row["frames"]) for row in receipts), "pending_videos": pending,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "runner_sha256": _sha256(Path(__file__)), "receipts": [f"extracted_{row['video_id']}.json" for row in receipts],
    })
    if pending:
        print(f"PARTIAL | {sum(len(row['frames']) for row in receipts)} frames ready; pending archives: {','.join(pending)}", flush=True)
    else:
        print("PROGRESS 100% | task manifest extracted and JPEG/XML dimensions verified", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pack", "pack-multi", "review", "validate", "task-extract"], default="pack")
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--reuse-run", type=Path, help="reuse identical task clips from an earlier extraction")
    parser.add_argument("--available-only", action="store_true", help="defer archives not yet promoted by acquisition")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--video-id", default="001-007")
    parser.add_argument("--minimum-gap", type=int, default=30)
    parser.add_argument("--anchor-frames", type=int, default=16)
    parser.add_argument("--pack-dir", type=Path)
    parser.add_argument("--annotation-archive", type=Path)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--review-manifest", type=Path)
    parser.add_argument("--pack-config", type=Path)
    args = parser.parse_args()
    if args.mode == "task-extract":
        if not args.task_manifest or not args.archive or not args.output_dir:
            parser.error("task-extract requires --task-manifest, --archive directory, --output-dir")
        extract_task_manifest(args.task_manifest.resolve(), args.archive.resolve(), args.output_dir.resolve(), args.run_dir.resolve(),
                              args.reuse_run.resolve() if args.reuse_run else None, args.available_only)
    elif args.mode == "pack":
        if not args.archive or not args.summary or not args.output_dir:
            parser.error("pack mode requires --archive, --summary, and --output-dir")
        run(
            [{"video_id": args.video_id, "archive_path": args.archive.resolve()}],
            args.summary.resolve(), args.output_dir.resolve(), args.run_dir.resolve(),
            args.minimum_gap, args.anchor_frames,
        )
    elif args.mode == "pack-multi":
        if not args.pack_config:
            parser.error("pack-multi mode requires --pack-config")
        pack_config_path = args.pack_config.resolve()
        pack_config = _load_multi_pack_config(pack_config_path)
        run(
            pack_config["archives"],
            Path(pack_config["summary_path"]).resolve(),
            Path(pack_config["output_dir"]).resolve(),
            args.run_dir.resolve(),
            int(pack_config["minimum_gap_frames"]),
            int(pack_config["anchor_frames"]),
            source_config_path=pack_config_path,
        )
    elif args.mode == "review":
        if not args.pack_dir or not args.annotation_archive or not args.review_dir:
            parser.error("review mode requires --pack-dir, --annotation-archive, and --review-dir")
        build_review(
            args.pack_dir.resolve(), args.annotation_archive.resolve(),
            args.review_dir.resolve(), args.run_dir.resolve(),
        )
    else:
        if not args.annotations or not args.review_manifest:
            parser.error("validate mode requires --annotations and --review-manifest")
        validate_review_annotations(
            args.annotations.resolve(), args.review_manifest.resolve(), args.run_dir.resolve()
        )
