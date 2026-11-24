"""Step 1: Dataset scanning, MediaPipe kinematic extraction, and cycle feature table generation.

Usage:
    python src/extract_features.py
    python src/extract_features.py --data-dir "видео/seriq-dataset" --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.biomechanics import (  # noqa: E402
    compute_cycle_features,
    ensure_mediapipe_models,
    extract_video_kinematics,
    segment_cycles,
)

DEFAULT_DATA_DIR = PROJECT_ROOT / "видео" / "seriq-dataset"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "e02_features.csv"
DEFAULT_OUTPUT_TIMESERIES = PROJECT_ROOT / "data" / "processed" / "e02_timeseries.json"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def is_e02_video(path: Path) -> bool:
    """Return True if a video path belongs to Exercise 02 (Extending the Elbow)."""
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return False
    parts_lower = [p.lower() for p in path.parts]
    if any(
        ("2_extending" in p)
        or ("extending the elbow" in p)
        or (p in {"e02", "exercise_02", "ex02"})
        for p in parts_lower
    ):
        return True
    tokens = path.stem.split("_")
    return len(tokens) >= 3 and tokens[1] in {"02", "2", "e02", "E02"}


def parse_video_metadata(path: Path) -> dict[str, Any]:
    """Parse `video_path`, `video_id`, `subject_id`, and `label` from directory/filename."""
    tokens = path.stem.split("_")
    subject_id = tokens[0] if len(tokens) >= 2 else "unknown"

    # Determine ground-truth class (`complete` -> 1, `incomplete` -> 0)
    # Folder label takes priority over filename token (handles Subject 04 `_00_` filename typo
    # inside `Complete/Train/04_02_00_*.mp4`).
    parts_lower = [p.lower() for p in path.parts[:-1]]
    folder_label: int | None = None
    if "incomplete" in parts_lower:
        folder_label = 0
    elif "complete" in parts_lower:
        folder_label = 1

    token_label: int | None = None
    if len(tokens) >= 3 and tokens[2] in {"00", "0", "01", "1"}:
        token_label = 1 if tokens[2] in {"01", "1"} else 0

    if folder_label is not None:
        label = folder_label
    elif token_label is not None:
        label = token_label
    else:
        raise ValueError(f"Cannot infer complete/incomplete class label from path: {path}")

    filename_mismatch = (
        folder_label is not None and token_label is not None and folder_label != token_label
    )
    label_name = "complete" if label == 1 else "incomplete"
    # Collision-free video_id (disambiguates 04_02_00_* in Complete vs Incomplete)
    video_id = f"S{subject_id}_{label_name}_{path.stem}"

    return {
        "video_path": str(path.resolve()),
        "video_id": video_id,
        "subject_id": str(subject_id),
        "label": int(label),
        "label_name": label_name,
        "filename_mismatch": bool(filename_mismatch),
    }


def scan_e02_dataset(data_dir: Path) -> list[dict[str, Any]]:
    """Scan dataset directory for E02 videos and print comprehensive summary statistics."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    e02_paths = sorted(p for p in data_dir.rglob("*") if p.is_file() and is_e02_video(p))
    if not e02_paths:
        raise RuntimeError(f"No E02 videos found under: {data_dir}")

    records = [parse_video_metadata(p) for p in e02_paths]

    n_total = len(records)
    subjects = sorted({r["subject_id"] for r in records})
    label_counts = Counter(r["label_name"] for r in records)
    mismatches = [r for r in records if r["filename_mismatch"]]

    print("=" * 72)
    print("SerIQ — Step 1: E02 (Elbow Extension) Dataset Summary")
    print("=" * 72)
    print(f"Dataset root         : {data_dir.resolve()}")
    print(f"Total E02 videos     : {n_total}")
    print(f"Unique volunteers    : {len(subjects)} -> {subjects}")
    print(
        f"Class balance        : complete (1) = {label_counts['complete']} "
        f"({100.0 * label_counts['complete'] / n_total:.1f}%), "
        f"incomplete (0) = {label_counts['incomplete']} "
        f"({100.0 * label_counts['incomplete'] / n_total:.1f}%)"
    )
    if mismatches:
        mismatch_subjects = sorted({r["subject_id"] for r in mismatches})
        print(
            f"[Data Hygiene Note]  : Detected {len(mismatches)} videos (subject(s) {mismatch_subjects}) "
            f"where folder class != filename token; resolved via directory ground truth and unique video_id."
        )

    print("\nPer-volunteer breakdown:")
    for sid in subjects:
        sub_recs = [r for r in records if r["subject_id"] == sid]
        c1 = sum(1 for r in sub_recs if r["label"] == 1)
        c0 = sum(1 for r in sub_recs if r["label"] == 0)
        print(f"  Volunteer {sid}: {len(sub_recs):3d} videos (complete={c1:2d}, incomplete={c0:2d})")
    print("=" * 72)

    return records


def _process_single_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Worker entrypoint for parallel per-video MediaPipe processing."""
    kin = extract_video_kinematics(rec["video_path"])
    if kin.get("status") != "success":
        return {
            "status": "error",
            "video_id": rec["video_id"],
            "error": kin.get("error", "Unknown extraction error"),
        }

    cycles = segment_cycles(kin["angle_smoothed"], kin["fps"])
    rows: list[dict[str, Any]] = []
    for cyc in cycles:
        feat = compute_cycle_features(cyc, kin)
        row: dict[str, Any] = {
            "video_id": rec["video_id"],
            "subject_id": rec["subject_id"],
            "cycle_idx": int(cyc["cycle_idx"]),
            "label": int(rec["label"]),
            "video_path": rec["video_path"],
            "backend": kin["backend"],
            "active_arm": kin["active_arm"],
            "fps": round(float(kin["fps"]), 3),
            "start_frame": int(cyc["start_frame"]),
            "peak_frame": int(cyc["peak_frame"]),
            "end_frame": int(cyc["end_frame"]),
        }
        row.update(feat)
        rows.append(row)

    # Compact downsampled curve for plotting (`sample_angle_curve.png`)
    step = max(1, len(kin["angle_smoothed"]) // 250)
    ts_payload = {
        "video_id": rec["video_id"],
        "subject_id": rec["subject_id"],
        "label": int(rec["label"]),
        "backend": kin["backend"],
        "active_arm": kin["active_arm"],
        "fps": round(float(kin["fps"]), 3),
        "times_sec": [round(float(t), 3) for t in kin["times_sec"][::step]],
        "angle_raw": [round(float(a), 2) for a in kin["angle_raw"][::step]],
        "angle_smoothed": [round(float(a), 2) for a in kin["angle_smoothed"][::step]],
        "cycles": cycles,
    }

    return {
        "status": "success",
        "video_id": rec["video_id"],
        "rows": rows,
        "timeseries": ts_payload,
    }


def run_extraction(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_csv: Path = DEFAULT_OUTPUT_CSV,
    output_timeseries: Path = DEFAULT_OUTPUT_TIMESERIES,
    workers: int = 6,
) -> pd.DataFrame:
    """Extract MediaPipe features for all E02 videos and save `e02_features.csv`."""
    ensure_mediapipe_models()
    records = scan_e02_dataset(data_dir)

    all_rows: list[dict[str, Any]] = []
    timeseries_store: dict[str, Any] = {}
    errors: list[tuple[str, str]] = []

    max_workers = max(1, min(workers, os.cpu_count() or 4))
    print(f"\nExtracting MediaPipe kinematics & segmenting cycles ({max_workers} workers)...")

    if max_workers == 1:
        for rec in tqdm(records, desc="E02 videos"):
            res = _process_single_record(rec)
            if res["status"] == "success":
                all_rows.extend(res["rows"])
                timeseries_store[res["video_id"]] = res["timeseries"]
            else:
                errors.append((res["video_id"], res["error"]))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_single_record, rec): rec for rec in records}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="E02 videos"):
                res = fut.result()
                if res["status"] == "success":
                    all_rows.extend(res["rows"])
                    timeseries_store[res["video_id"]] = res["timeseries"]
                else:
                    errors.append((res["video_id"], res["error"]))

    if not all_rows:
        raise RuntimeError("No cycles could be extracted from E02 dataset.")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["subject_id", "label", "video_id", "cycle_idx"]).reset_index(drop=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    with open(output_timeseries, "w", encoding="utf-8") as f:
        json.dump(timeseries_store, f, ensure_ascii=False)

    print("\nExtraction complete!")
    print(f"Processed videos     : {len(timeseries_store)} / {len(records)}")
    if errors:
        print(f"Skipped/failed videos: {len(errors)}")
        for vid, err in errors[:5]:
            print(f"  - {vid}: {err}")
    print(f"Total extracted cycles: {len(df)}")
    print(
        f"Cycle class balance  : complete (1) = {int((df['label'] == 1).sum())}, "
        f"incomplete (0) = {int((df['label'] == 0).sum())}"
    )
    print(
        f"Max angle by class   : complete mean={df.loc[df['label']==1, 'max_angle'].mean():.1f}° "
        f"(±{df.loc[df['label']==1, 'max_angle'].std():.1f}°), "
        f"incomplete mean={df.loc[df['label']==0, 'max_angle'].mean():.1f}° "
        f"(±{df.loc[df['label']==0, 'max_angle'].std():.1f}°)"
    )
    print(f"Saved feature table  : {output_csv}")
    print(f"Saved time series    : {output_timeseries}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MediaPipe E02 features.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root directory containing the rehabilitation video dataset.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Destination path for extracted cycle features CSV.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of parallel worker processes.",
    )
    args = parser.parse_args()
    run_extraction(data_dir=args.data_dir, output_csv=args.output_csv, workers=args.workers)


if __name__ == "__main__":
    main()
