"""Step 3: Production inference module for FastAPI / Streamlit integration (`predict_video`).

Exposes a clean single-call interface:
    from src.inference import predict_video
    result = predict_video("path/to/video.mp4")

CLI usage:
    python src/inference.py "path/to/video.mp4"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.biomechanics import (  # noqa: E402
    compute_cycle_features,
    extract_video_kinematics,
    segment_cycles,
)
from src.model_bundle import CalibratedModelBundle  # noqa: F401, E402

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
_INFERENCE_CACHE: dict[str, Any] = {}


def _load_artifacts(artifacts_dir: Path = ARTIFACTS_DIR) -> tuple[Any, dict[str, Any]]:
    """Lazy-load and cache trained model bundle and configuration JSON."""
    cache_key = str(artifacts_dir.resolve())
    if cache_key not in _INFERENCE_CACHE:
        model_path = artifacts_dir / "best_model.joblib"
        config_path = artifacts_dir / "config.json"
        if not model_path.exists() or not config_path.exists():
            raise FileNotFoundError(
                f"Trained model artifacts not found in {artifacts_dir}. "
                f"Run `python src/train.py` before calling `predict_video`."
            )
        model_bundle = joblib.load(model_path)
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        _INFERENCE_CACHE[cache_key] = (model_bundle, config)
    return _INFERENCE_CACHE[cache_key]


def _error_response() -> dict[str, Any]:
    """Return schema-compliant error dictionary when person/arm is missing or unreadable."""
    return {
        "status": "error",
        "total_cycles": 0,
        "completed_cycles": 0,
        "review_cycles": 0,
        "cycles": [],
        "angle_series": [],
    }


def predict_video(video_path: str) -> dict[str, Any]:
    """Evaluate elbow extension exercise (E02) repetitions on a video file.

    Args:
        video_path: Path to an uploaded or webcam-recorded video file.

    Returns:
        Strictly formatted dictionary ready for FastAPI / Streamlit frontend consumption:
        {
            "status": "success" | "error",
            "total_cycles": int,
            "completed_cycles": int,
            "review_cycles": int,
            "cycles": [
                {
                    "cycle_id": int,
                    "raw_class": "complete" | "incomplete",
                    "ui_status": "Выполнено" | "Проверьте инструкцию",
                    "confidence": float,
                    "max_angle": float,
                    "min_angle": float,
                }
            ],
            "angle_series": [
                {"frame": int, "time_sec": float, "angle": float},
                ...
            ]
        }
    """
    model_bundle, config = _load_artifacts()
    threshold = float(config.get("confidence_threshold", 0.75))
    ui_labels = config.get(
        "ui_status_mapping",
        {"completed": "Выполнено", "review": "Проверьте инструкцию"},
    )

    kinematics = extract_video_kinematics(video_path)
    if kinematics.get("status") != "success":
        return _error_response()

    angle_smoothed = kinematics["angle_smoothed"]
    times_sec = kinematics["times_sec"]
    fps = float(kinematics["fps"])

    angle_series = [
        {
            "frame": int(frame_idx),
            "time_sec": round(float(t_sec), 3),
            "angle": round(float(ang), 1),
        }
        for frame_idx, (t_sec, ang) in enumerate(zip(times_sec, angle_smoothed))
    ]

    detected_cycles = segment_cycles(angle_smoothed, fps)
    if not detected_cycles:
        return _error_response()

    feature_rows = [compute_cycle_features(cyc, kinematics) for cyc in detected_cycles]
    X_df = pd.DataFrame(feature_rows)
    probs_complete = model_bundle.predict_proba(X_df)[:, 1]

    cycles_output: list[dict[str, Any]] = []
    completed_count = 0
    review_count = 0

    for idx, (cyc, prob_c) in enumerate(zip(detected_cycles, probs_complete), start=1):
        p_complete = float(prob_c)
        raw_class = "complete" if p_complete >= 0.5 else "incomplete"
        confidence = p_complete if raw_class == "complete" else (1.0 - p_complete)

        if raw_class == "complete" and confidence >= threshold:
            ui_status = str(ui_labels["completed"])
            completed_count += 1
        else:
            ui_status = str(ui_labels["review"])
            review_count += 1

        cycles_output.append(
            {
                "cycle_id": int(idx),
                "raw_class": raw_class,
                "ui_status": ui_status,
                "confidence": round(float(confidence), 4),
                "max_angle": round(float(cyc["max_angle"]), 1),
                "min_angle": round(float(cyc["min_angle"]), 1),
            }
        )

    return {
        "status": "success",
        "total_cycles": int(len(cycles_output)),
        "completed_cycles": int(completed_count),
        "review_cycles": int(review_count),
        "cycles": cycles_output,
        "angle_series": angle_series,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SerIQ E02 inference on a video file.")
    parser.add_argument("video_path", type=str, help="Path to input video (.mp4, .avi, .webm).")
    parser.add_argument(
        "--max-series-preview",
        type=int,
        default=5,
        help="Number of angle_series points to preview in console output (0 for full JSON).",
    )
    args = parser.parse_args()

    result = predict_video(args.video_path)
    if args.max_series_preview > 0 and len(result["angle_series"]) > args.max_series_preview:
        preview = dict(result)
        preview["angle_series"] = (
            result["angle_series"][: args.max_series_preview]
            + [{"note": f"... ({len(result['angle_series'])} total frames) ..."}]
        )
        print(json.dumps(preview, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
