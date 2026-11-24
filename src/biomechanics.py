"""Shared MediaPipe biomechanics extraction, smoothing, and cycle segmentation for E02.

Supports two complementary MediaPipe vision backends out of the box:
1. Primary (`PoseLandmarker`): extracts shoulder (11/12), elbow (13/14), wrist (15/16),
   and hip (23/24) landmarks with visibility gating on standard upper-body / webcam videos.
2. Fallback (`ImageSegmenter` - SelfieSegmenter): automatically activates on privacy-cropped
   clinical close-up videos where the head/torso are outside the camera frame, reconstructing
   forearm kinematics on a calibrated anatomical scale.

Models are loaded into memory via `model_asset_buffer` (`bytes`), ensuring full compatibility
with non-ASCII / Cyrillic filesystem paths on Windows.
"""

from __future__ import annotations

import os
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from scipy.signal import find_peaks, medfilt, savgol_filter

warnings.filterwarnings("ignore", category=UserWarning, module="mediapipe")
cv2.setNumThreads(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
SEGMENTER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)

# MediaPipe Pose landmark indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24

_MODEL_CACHE: dict[str, bytes] = {}
_INSTANCE_CACHE: dict[str, Any] = {}


def _get_image_segmenter(seg_model_bytes: bytes) -> vision.ImageSegmenter:
    """Return process-cached MediaPipe ImageSegmenter instance."""
    if "segmenter" not in _INSTANCE_CACHE:
        opts = vision.ImageSegmenterOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=seg_model_bytes),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
        )
        _INSTANCE_CACHE["segmenter"] = vision.ImageSegmenter.create_from_options(opts)
    return _INSTANCE_CACHE["segmenter"]


def ensure_mediapipe_models(models_dir: Path = MODELS_DIR) -> tuple[bytes, bytes]:
    """Ensure local MediaPipe `.task`/`.tflite` bundles exist and return raw bytes."""
    models_dir.mkdir(parents=True, exist_ok=True)
    pose_path = models_dir / "pose_landmarker_lite.task"
    seg_path = models_dir / "selfie_segmenter.tflite"

    if "pose" not in _MODEL_CACHE:
        if not pose_path.exists():
            urllib.request.urlretrieve(POSE_MODEL_URL, pose_path)
        _MODEL_CACHE["pose"] = pose_path.read_bytes()

    if "seg" not in _MODEL_CACHE:
        if not seg_path.exists():
            urllib.request.urlretrieve(SEGMENTER_MODEL_URL, seg_path)
        _MODEL_CACHE["seg"] = seg_path.read_bytes()

    return _MODEL_CACHE["pose"], _MODEL_CACHE["seg"]


def compute_joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Compute internal joint angle ABC (in degrees, [0, 180]) at vertex B."""
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return float("nan")
    cos_theta = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_theta)))


def despike_and_smooth_angles(
    sampled_frames: np.ndarray,
    sampled_angles: np.ndarray,
    total_frames: int,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate valid angle samples to all frames and apply median + Savitzky-Golay smoothing."""
    valid_mask = np.isfinite(sampled_angles)
    if int(np.sum(valid_mask)) < 5:
        raise ValueError("Insufficient valid joint angle observations in video.")

    valid_idx = sampled_frames[valid_mask].astype(np.float64)
    valid_ang = sampled_angles[valid_mask].astype(np.float64)

    # 1. Remove isolated detection spikes via rolling median residual gate (~0.6s window)
    if len(valid_ang) >= 9:
        k_med = 9 if len(valid_ang) >= 15 else 5
        baseline = medfilt(valid_ang, kernel_size=k_med)
        residual = np.abs(valid_ang - baseline)
        mad = float(np.median(residual)) + 1e-3
        inlier = residual <= max(14.0, 3.8 * mad)
        if int(np.sum(inlier)) >= 6:
            valid_idx = valid_idx[inlier]
            valid_ang = valid_ang[inlier]

    # 2. Interpolate onto full native video frame grid [0 .. total_frames - 1]
    all_frames = np.arange(total_frames, dtype=np.float64)
    raw_full = np.interp(all_frames, valid_idx, valid_ang)

    # 3. Median despike (~0.3s) + Savitzky-Golay smoothing (~0.7s window)
    med_k = int(round(max(5, fps * 0.30)))
    if med_k % 2 == 0:
        med_k += 1
    if total_frames > med_k:
        raw_despiked = medfilt(raw_full, kernel_size=med_k)
    else:
        raw_despiked = raw_full.copy()

    win = int(round(max(9, fps * 0.70)))
    if win % 2 == 0:
        win += 1
    win = min(win, total_frames if total_frames % 2 == 1 else total_frames - 1)

    if win >= 5:
        poly = min(3, win - 2)
        smoothed = savgol_filter(raw_despiked, window_length=win, polyorder=poly, mode="interp")
    else:
        smoothed = raw_despiked.copy()

    smoothed = np.clip(smoothed, 0.0, 180.0)
    return raw_full, smoothed


def _extract_with_pose_landmarker(
    frames_bgr: list[np.ndarray],
    sampled_indices: np.ndarray,
    fps: float,
    pose_model_bytes: bytes,
) -> dict[str, Any]:
    """Run MediaPipe PoseLandmarker (VIDEO mode) and evaluate active arm & upper-body quality."""
    n_s = len(frames_bgr)
    h, w = frames_bgr[0].shape[:2]
    diag = float(np.hypot(w, h))

    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=pose_model_bytes),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.25,
        min_pose_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )

    coords: dict[str, np.ndarray] = {
        k: np.full((n_s, 2), np.nan, dtype=np.float64)
        for k in ("l_sh", "l_el", "l_wr", "l_hip", "r_sh", "r_el", "r_wr", "r_hip")
    }
    vis: dict[str, np.ndarray] = {
        k: np.zeros(n_s, dtype=np.float64)
        for k in ("l_arm", "r_arm", "l_hip", "r_hip")
    }
    head_above_shoulder = np.zeros(n_s, dtype=bool)
    early_check_idx = min(75, max(25, int(0.45 * n_s)))

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for i, (frame_idx, bgr) in enumerate(zip(sampled_indices, frames_bgr)):
            # Early exit on close-up forearm crops (where no head/shoulder anatomy is in frame)
            if i == early_check_idx and int(np.sum(head_above_shoulder[:i])) < 3:
                return {"reliable": False, "reason": "No upper-body head/shoulder context."}

            ts_ms = int(round(float(frame_idx) * 1000.0 / max(fps, 1.0)))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = landmarker.detect_for_video(mp_img, ts_ms)
            if not res.pose_landmarks:
                continue
            lm = res.pose_landmarks[0]

            sh_mid_y = 0.5 * (lm[LEFT_SHOULDER].y + lm[RIGHT_SHOULDER].y)
            if (sh_mid_y - lm[NOSE].y) >= 0.08 and 0.02 <= lm[NOSE].y <= 0.70:
                head_above_shoulder[i] = True

            for key, idx in (
                ("l_sh", LEFT_SHOULDER),
                ("l_el", LEFT_ELBOW),
                ("l_wr", LEFT_WRIST),
                ("l_hip", LEFT_HIP),
                ("r_sh", RIGHT_SHOULDER),
                ("r_el", RIGHT_ELBOW),
                ("r_wr", RIGHT_WRIST),
                ("r_hip", RIGHT_HIP),
            ):
                coords[key][i] = [lm[idx].x * w, lm[idx].y * h]

            vis["l_arm"][i] = min(
                lm[LEFT_SHOULDER].visibility,
                lm[LEFT_ELBOW].visibility,
                lm[LEFT_WRIST].visibility,
            )
            vis["r_arm"][i] = min(
                lm[RIGHT_SHOULDER].visibility,
                lm[RIGHT_ELBOW].visibility,
                lm[RIGHT_WRIST].visibility,
            )
            vis["l_hip"][i] = lm[LEFT_HIP].visibility
            vis["r_hip"][i] = lm[RIGHT_HIP].visibility

    if float(np.mean(head_above_shoulder)) < 0.25:
        return {"reliable": False, "reason": "Insufficient upper-body presence across video."}

    # Evaluate each arm's validity and active wrist coordinate variance
    side_stats: dict[str, dict[str, Any]] = {}
    for side, pfx in (("left", "l"), ("right", "r")):
        sh = coords[f"{pfx}_sh"]
        el = coords[f"{pfx}_el"]
        wr = coords[f"{pfx}_wr"]
        hp = coords[f"{pfx}_hip"]
        v_arm = vis[f"{pfx}_arm"]
        v_hip = vis[f"{pfx}_hip"]

        upper_len = np.linalg.norm(sh - el, axis=1)
        fore_len = np.linalg.norm(wr - el, axis=1)
        in_bounds = (
            head_above_shoulder
            & (sh[:, 0] >= -0.05 * w)
            & (sh[:, 0] <= 1.05 * w)
            & (sh[:, 1] >= 0.10 * h)
            & (sh[:, 1] <= 0.92 * h)
        )
        plausible = (
            (v_arm >= 0.45)
            & in_bounds
            & (upper_len >= 0.07 * diag)
            & (fore_len >= 0.07 * diag)
            & (upper_len <= 0.65 * diag)
            & (fore_len <= 0.65 * diag)
        )
        valid_count = int(np.sum(plausible))
        valid_ratio = valid_count / max(n_s, 1)
        if valid_count >= 5:
            wr_valid = wr[plausible]
            wrist_var = float(np.var(wr_valid[:, 0]) + np.var(wr_valid[:, 1]))
            activity_score = wrist_var * (valid_ratio ** 1.2) * float(np.mean(v_arm[plausible]))
        else:
            wrist_var = 0.0
            activity_score = 0.0

        side_stats[side] = {
            "prefix": pfx,
            "plausible": plausible,
            "valid_count": valid_count,
            "valid_ratio": valid_ratio,
            "wrist_var": wrist_var,
            "activity_score": activity_score,
            "mean_vis": float(np.mean(v_arm[plausible])) if valid_count > 0 else 0.0,
            "sh": sh,
            "el": el,
            "wr": wr,
            "hp": hp,
            "v_arm": v_arm,
            "v_hip": v_hip,
            "fore_len": fore_len,
        }

    # Check if camera is in lateral/profile view (shoulders overlap horizontally < 0.16 * w).
    # In side-profile videos (especially with blurred/occluded faces), BlazePose may flip
    # left/right chirality mid-video while localizing the same foreground arm; selecting the
    # higher-visibility chirality per frame preserves continuous foreground arm tracking.
    both_det = np.isfinite(coords["l_sh"][:, 0]) & np.isfinite(coords["r_sh"][:, 0])
    is_lateral_view = (
        int(np.sum(both_det)) >= 5
        and float(np.nanmedian(np.abs(coords["l_sh"][both_det, 0] - coords["r_sh"][both_det, 0])))
        < 0.16 * w
    )

    active_side = (
        "right"
        if side_stats["right"]["activity_score"] >= side_stats["left"]["activity_score"]
        else "left"
    )

    if is_lateral_view:
        sh_f = np.full((n_s, 2), np.nan, dtype=np.float64)
        el_f = np.full((n_s, 2), np.nan, dtype=np.float64)
        wr_f = np.full((n_s, 2), np.nan, dtype=np.float64)
        hp_f = np.full((n_s, 2), np.nan, dtype=np.float64)
        va_f = np.zeros(n_s, dtype=np.float64)
        vh_f = np.zeros(n_s, dtype=np.float64)
        pl_f = np.zeros(n_s, dtype=bool)
        fl_f = np.full(n_s, np.nan, dtype=np.float64)

        r_st, l_st = side_stats["right"], side_stats["left"]
        for i in range(n_s):
            use_r = (r_st["plausible"][i] and not l_st["plausible"][i]) or (
                r_st["plausible"][i]
                and l_st["plausible"][i]
                and r_st["v_arm"][i] >= l_st["v_arm"][i]
            )
            use_l = l_st["plausible"][i] and not use_r
            src = r_st if use_r else (l_st if use_l else None)
            if src is not None:
                pl_f[i] = True
                sh_f[i], el_f[i], wr_f[i], hp_f[i] = (
                    src["sh"][i],
                    src["el"][i],
                    src["wr"][i],
                    src["hp"][i],
                )
                va_f[i], vh_f[i], fl_f[i] = src["v_arm"][i], src["v_hip"][i], src["fore_len"][i]

        vc = int(np.sum(pl_f))
        vr = vc / max(n_s, 1)
        wv = float(np.var(wr_f[pl_f, 0]) + np.var(wr_f[pl_f, 1])) if vc >= 5 else 0.0
        mv = float(np.mean(va_f[pl_f])) if vc > 0 else 0.0
        chosen = {
            "plausible": pl_f,
            "valid_ratio": vr,
            "wrist_var": wv,
            "mean_vis": mv,
            "sh": sh_f,
            "el": el_f,
            "wr": wr_f,
            "hp": hp_f,
            "v_arm": va_f,
            "v_hip": vh_f,
            "fore_len": fl_f,
        }
    else:
        chosen = side_stats[active_side]

    is_reliable_pose = (
        chosen["valid_ratio"] >= 0.25
        and chosen["mean_vis"] >= 0.45
        and chosen["wrist_var"] >= (0.010 * diag) ** 2
    )
    if not is_reliable_pose:
        return {"reliable": False, "side_stats": side_stats}

    mask = chosen["plausible"]
    angles = np.full(n_s, np.nan, dtype=np.float64)
    trunk_angles = np.full(n_s, np.nan, dtype=np.float64)
    for i in np.where(mask)[0]:
        angles[i] = compute_joint_angle_deg(chosen["sh"][i], chosen["el"][i], chosen["wr"][i])
        if chosen["v_hip"][i] >= 0.35 and np.all(np.isfinite(chosen["hp"][i])):
            trunk_angles[i] = compute_joint_angle_deg(
                chosen["hp"][i], chosen["sh"][i], chosen["el"][i]
            )

    med_forearm = float(np.nanmedian(chosen["fore_len"][mask]))
    sh_valid = chosen["sh"][mask]
    sh_disp = float(
        np.hypot(np.std(sh_valid[:, 0]), np.std(sh_valid[:, 1])) / max(med_forearm, 1.0)
    )
    trunk_std = (
        float(np.nanstd(trunk_angles)) if int(np.sum(np.isfinite(trunk_angles))) >= 5 else 0.0
    )

    return {
        "reliable": True,
        "backend": "mediapipe_pose",
        "active_arm": active_side,
        "sampled_angles": angles,
        "mean_visibility": chosen["mean_vis"],
        "valid_ratio": chosen["valid_ratio"],
        "wrist_variance": chosen["wrist_var"],
        "shoulder_disp_norm": sh_disp,
        "trunk_angle_std": trunk_std,
    }


def _extract_with_limb_segmenter(
    frames_bgr: list[np.ndarray],
    seg_model_bytes: bytes,
) -> dict[str, Any]:
    """MediaPipe ImageSegmenter forearm tracker for close-up limb-only videos."""
    n_s = len(frames_bgr)
    seg_w, seg_h = 288, 162
    conf_maps = np.zeros((n_s, seg_h, seg_w), dtype=np.float32)

    segmenter = _get_image_segmenter(seg_model_bytes)
    for i, bgr in enumerate(frames_bgr):
        small_bgr = cv2.resize(bgr, (seg_w, seg_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = segmenter.segment(mp_img)
        conf_maps[i] = res.confidence_masks[0].numpy_view().squeeze()

    h, w = seg_h, seg_w

    bin_masks = (conf_maps >= 0.25).astype(np.uint8)
    # Clean masks and keep primary limb component
    cleaned_masks: list[np.ndarray] = []
    areas: list[int] = []
    kernel = np.ones((5, 5), np.uint8)
    for m in bin_masks:
        mc = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        mc = cv2.morphologyEx(mc, cv2.MORPH_CLOSE, kernel)
        num_l, labels, stats, _ = cv2.connectedComponentsWithStats(mc)
        if num_l > 1:
            best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            area = int(stats[best, cv2.CC_STAT_AREA])
            if area >= int(0.012 * h * w):
                mc = np.where(labels == best, 1, 0).astype(np.uint8)
                cleaned_masks.append(mc)
                areas.append(area)
                continue
        cleaned_masks.append(np.zeros_like(mc))
        areas.append(0)

    valid_mask_frames = sum(1 for a in areas if a > 0)
    if valid_mask_frames < max(6, int(0.40 * n_s)):
        return {"reliable": False, "reason": "No person or active arm detected in frame."}

    # Check temporal motion so a completely static scene / inanimate object is rejected
    stack_gray = np.array(
        [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr], dtype=np.float32
    )
    motion_map = np.std(stack_gray, axis=0)
    moving_ratio = float(np.mean(motion_map > 6.0))
    if moving_ratio < 0.008:
        return {"reliable": False, "reason": "No limb motion detected in video."}

    freq = np.mean(cleaned_masks, axis=0)
    ys, xs = np.where(freq >= 0.45)
    all_ys, all_xs = np.where(freq >= 0.05)
    if len(xs) < 40 or len(all_xs) < 60:
        return {"reliable": False, "reason": "Insufficient limb foreground consistency."}

    # Determine extension direction (rightward vs leftward from stationary elbow anchor)
    span_right = float(np.percentile(all_xs, 92) - np.percentile(xs, 50))
    span_left = float(np.percentile(xs, 50) - np.percentile(all_xs, 8))
    facing_right = span_right >= span_left

    bot_band = ys >= np.percentile(ys, 75)
    elbow_y = float(np.percentile(ys, 92))
    if facing_right:
        elbow_x = float(np.percentile(xs[bot_band], 28))
        sh_vec = np.array([-np.cos(np.radians(26.0)), -np.sin(np.radians(26.0))], dtype=np.float64)
        active_side = "right"
    else:
        elbow_x = float(np.percentile(xs[bot_band], 72))
        sh_vec = np.array([np.cos(np.radians(26.0)), -np.sin(np.radians(26.0))], dtype=np.float64)
        active_side = "left"

    elbow = np.array([elbow_x, elbow_y], dtype=np.float64)

    wrist_pts = np.full((n_s, 2), np.nan, dtype=np.float64)
    fore_lengths = np.full(n_s, np.nan, dtype=np.float64)
    vis_scores = np.zeros(n_s, dtype=np.float64)

    for i, (m, cm) in enumerate(zip(cleaned_masks, conf_maps)):
        # If confidence mask dipped when arm lay flat on dark table, relax threshold locally
        cur_m = m
        if areas[i] < 0.55 * float(np.median([a for a in areas if a > 0])):
            relaxed = (cm >= 0.12).astype(np.uint8)
            relaxed = cv2.morphologyEx(relaxed, cv2.MORPH_OPEN, kernel)
            num_l, labels, stats, _ = cv2.connectedComponentsWithStats(relaxed)
            if num_l > 1:
                best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                if stats[best, cv2.CC_STAT_AREA] >= int(0.012 * h * w):
                    cur_m = np.where(labels == best, 1, 0).astype(np.uint8)

        my, mx = np.where(cur_m > 0)
        if facing_right:
            keep = mx >= elbow_x - 6
        else:
            keep = mx <= elbow_x + 6
        mx, my = mx[keep], my[keep]
        if len(mx) < 25:
            continue

        pts = np.column_stack([mx, my]).astype(np.float64)
        dists = np.linalg.norm(pts - elbow, axis=1)
        top_cut = np.percentile(dists, 86)
        wrist = pts[dists >= top_cut].mean(axis=0)
        flen = float(np.linalg.norm(wrist - elbow))
        wrist_pts[i] = wrist
        fore_lengths[i] = flen
        vis_scores[i] = float(np.mean(cm[cur_m > 0]))

    valid_len = fore_lengths[np.isfinite(fore_lengths)]
    if len(valid_len) < 6:
        return {"reliable": False, "reason": "Could not track distal wrist trajectory."}

    med_flen = float(np.median(valid_len))
    # Reject foreshortened outlier frames where hand segmentation momentarily clipped
    good = np.isfinite(fore_lengths) & (fore_lengths >= 0.58 * med_flen)
    if int(np.sum(good)) < 6:
        return {"reliable": False, "reason": "Unstable forearm tracking length."}

    angles = np.full(n_s, np.nan, dtype=np.float64)
    for i in np.where(good)[0]:
        vf = wrist_pts[i] - elbow
        vf_norm = float(np.linalg.norm(vf))
        if vf_norm > 1e-6:
            cos_a = float(np.clip(np.dot(sh_vec, vf / vf_norm), -1.0, 1.0))
            angles[i] = float(np.degrees(np.arccos(cos_a)))

    wr_good = wrist_pts[good]
    wrist_var = float(np.var(wr_good[:, 0]) + np.var(wr_good[:, 1]))
    if wrist_var < (0.015 * np.hypot(w, h)) ** 2:
        return {"reliable": False, "reason": "Wrist dispersion too small (no exercise movement)."}

    return {
        "reliable": True,
        "backend": "mediapipe_limb_segmenter",
        "active_arm": active_side,
        "sampled_angles": angles,
        "mean_visibility": float(np.mean(vis_scores[good])),
        "valid_ratio": float(np.mean(good)),
        "wrist_variance": wrist_var,
        "shoulder_disp_norm": 0.0,
        "trunk_angle_std": 0.0,
    }


def extract_video_kinematics(
    video_path: str | Path,
    target_sample_fps: float = 15.0,
    proc_width: int = 480,
) -> dict[str, Any]:
    """Process a video file frame-by-frame with OpenCV & MediaPipe and extract theta(t)."""
    path = Path(video_path)
    if not path.exists():
        return {"status": "error", "error": f"Video file not found: {path}"}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"status": "error", "error": f"Failed to open video stream: {path}"}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0 or fps > 240.0 or np.isnan(fps):
        fps = 30.0

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    stride = max(1, int(round(fps / target_sample_fps)))

    sampled_frames_bgr: list[np.ndarray] = []
    sampled_indices_list: list[int] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame_idx % stride == 0:
            h0, w0 = frame.shape[:2]
            if w0 > proc_width:
                scale = proc_width / float(w0)
                proc_h = max(64, int(round(h0 * scale)))
                small = cv2.resize(frame, (proc_width, proc_h), interpolation=cv2.INTER_AREA)
            else:
                small = frame
            sampled_frames_bgr.append(small)
            sampled_indices_list.append(frame_idx)
        frame_idx += 1

    cap.release()
    total_frames = frame_idx

    if total_frames < 10 or len(sampled_frames_bgr) < 6:
        return {"status": "error", "error": "Video is too short or contains no readable frames."}

    sampled_indices = np.array(sampled_indices_list, dtype=np.int64)
    pose_bytes, seg_bytes = ensure_mediapipe_models()

    # 1. Primary MediaPipe PoseLandmarker extraction
    pose_res = _extract_with_pose_landmarker(sampled_frames_bgr, sampled_indices, fps, pose_bytes)

    # 2. Automatic fallback to MediaPipe ImageSegmenter for close-up cropped limb videos
    if pose_res.get("reliable", False):
        chosen_res = pose_res
    else:
        seg_res = _extract_with_limb_segmenter(sampled_frames_bgr, seg_bytes)
        if not seg_res.get("reliable", False):
            return {
                "status": "error",
                "error": seg_res.get("reason", "Person or active arm not detected in frame."),
            }
        chosen_res = seg_res

    try:
        angle_raw, angle_smoothed = despike_and_smooth_angles(
            sampled_indices,
            chosen_res["sampled_angles"],
            total_frames=total_frames,
            fps=fps,
        )
    except ValueError as exc:
        return {"status": "error", "error": str(exc)}

    times_sec = np.arange(total_frames, dtype=np.float64) / fps

    return {
        "status": "success",
        "video_path": str(path),
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": float(total_frames / fps),
        "resolution": (orig_w, orig_h),
        "backend": chosen_res["backend"],
        "active_arm": chosen_res["active_arm"],
        "mean_visibility": float(chosen_res["mean_visibility"]),
        "valid_ratio": float(chosen_res["valid_ratio"]),
        "wrist_variance": float(chosen_res["wrist_variance"]),
        "shoulder_disp_norm": float(chosen_res["shoulder_disp_norm"]),
        "trunk_angle_std": float(chosen_res["trunk_angle_std"]),
        "times_sec": times_sec,
        "angle_raw": angle_raw,
        "angle_smoothed": angle_smoothed,
    }


def segment_cycles(
    angle_smoothed: np.ndarray,
    fps: float,
    min_prominence_deg: float = 14.0,
    min_extension_rise_deg: float = 12.0,
    min_cycle_sec: float = 2.2,
) -> list[dict[str, Any]]:
    """Segment elbow angle time series theta(t) into repetition cycles using `find_peaks`.

    Each cycle represents an active flexion -> peak extension (-> flexion return / plateau)
    movement. Pre-exercise static extension at frame 0 without a preceding flexion trough
    and sub-second tracking tremors are explicitly filtered out.
    """
    n = len(angle_smoothed)
    if n < max(10, int(fps * 1.0)):
        return []

    min_dist = max(8, int(round(fps * min_cycle_sec)))

    # 1. Interior extension peaks and flexion troughs via scipy.signal.find_peaks
    peaks, _ = find_peaks(angle_smoothed, distance=min_dist, prominence=min_prominence_deg)
    troughs, _ = find_peaks(-angle_smoothed, distance=max(6, int(round(fps * 1.4))), prominence=8.0)

    trough_set = sorted(set([0] + [int(t) for t in troughs] + [n - 1]))
    peak_candidates = [int(p) for p in peaks]

    # 2. Check for terminal extension plateau (when trial ends while holding full extension on table)
    prior_troughs = [t for t in troughs if t < n - int(round(0.8 * fps))]
    if prior_troughs:
        last_interior_trough = int(prior_troughs[-1])
    else:
        search_end = max(4, int(round(0.75 * n)))
        last_interior_trough = int(np.argmin(angle_smoothed[:search_end]))

    peaks_after_trough = [p for p in peak_candidates if p > last_interior_trough]
    if not peaks_after_trough and last_interior_trough < n - max(6, int(round(0.75 * fps))):
        tail = angle_smoothed[last_interior_trough:]
        rel_peak = int(np.argmax(tail))
        cand_peak = last_interior_trough + rel_peak
        if float(angle_smoothed[cand_peak] - angle_smoothed[last_interior_trough]) >= min_extension_rise_deg:
            peak_candidates.append(cand_peak)

    peak_candidates = sorted(set(peak_candidates))

    # 3. Pair each extension peak with its bounding flexion troughs [start_frame, peak_frame, end_frame]
    cycles: list[dict[str, Any]] = []
    min_ext_frames = max(6, int(round(0.75 * fps)))
    min_cyc_frames = max(12, int(round(1.50 * fps)))

    for p in peak_candidates:
        if p < min_ext_frames:
            continue

        prev_bound = cycles[-1]["end_frame"] if cycles else 0
        left_troughs = [t for t in trough_set if prev_bound <= t < p]
        if left_troughs and cycles:
            t_left = left_troughs[-1]
            win_start = (
                left_troughs[-2]
                if len(left_troughs) >= 2 and (p - t_left) < min_ext_frames
                else t_left
            )
            s = win_start + int(np.argmin(angle_smoothed[win_start : p + 1]))
        else:
            s = prev_bound + int(np.argmin(angle_smoothed[prev_bound : p + 1]))

        ext_rise = float(angle_smoothed[p] - angle_smoothed[s])
        if ext_rise < min_extension_rise_deg or (p - s) < min_ext_frames:
            continue

        right_troughs = [t for t in trough_set if t > p]
        next_peaks = [npk for npk in peak_candidates if npk > p]
        upper_lim = next_peaks[0] if next_peaks else (n - 1)
        valid_right = [t for t in right_troughs if t <= upper_lim]
        if valid_right:
            e = valid_right[0]
        else:
            e = p + int(np.argmin(angle_smoothed[p : upper_lim + 1]))
            if e <= p + 4:
                e = upper_lim

        if (e - s) < min_cyc_frames:
            continue

        seg = angle_smoothed[s : e + 1]
        cycles.append(
            {
                "cycle_idx": len(cycles) + 1,
                "start_frame": int(s),
                "peak_frame": int(p),
                "end_frame": int(e),
                "start_sec": float(s / fps),
                "peak_sec": float(p / fps),
                "end_sec": float(e / fps),
                "max_angle": float(np.max(seg)),
                "min_angle": float(np.min(seg)),
                "start_angle": float(angle_smoothed[s]),
                "end_angle": float(angle_smoothed[e]),
            }
        )

    # 4. Fallback if mild incomplete movement had prominence slightly below find_peaks threshold
    if not cycles:
        search_end = max(4, int(round(0.80 * n)))
        s = int(np.argmin(angle_smoothed[:search_end]))
        if s < n - min_ext_frames:
            p = s + int(np.argmax(angle_smoothed[s:]))
            if float(angle_smoothed[p] - angle_smoothed[s]) >= 6.0 and (p - s) >= max(4, int(round(0.45 * fps))):
                e = n - 1
                seg = angle_smoothed[s : e + 1]
                cycles.append(
                    {
                        "cycle_idx": 1,
                        "start_frame": int(s),
                        "peak_frame": int(p),
                        "end_frame": int(e),
                        "start_sec": float(s / fps),
                        "peak_sec": float(p / fps),
                        "end_sec": float(e / fps),
                        "max_angle": float(np.max(seg)),
                        "min_angle": float(np.min(seg)),
                        "start_angle": float(angle_smoothed[s]),
                        "end_angle": float(angle_smoothed[e]),
                    }
                )

    return cycles


def compute_cycle_features(
    cycle: dict[str, Any],
    kinematics: dict[str, Any],
    seq_len: int = 64,
    ext_seq_len: int = 16,
) -> dict[str, float]:
    """Compute biomechanical & resampled sequence features for a single repetition cycle.

    Core predictive features focus on peak amplitude, ROM, extension-phase kinematics
    [start_frame -> peak_frame], smoothness/jerk, and normalized extension trajectory so
    predictions remain consistent across both single-rep clinical trials and continuous
    multi-rep webcam sessions.
    """
    ang = kinematics["angle_smoothed"]
    fps = float(kinematics["fps"])
    dt = 1.0 / max(fps, 1.0)

    s = int(cycle["start_frame"])
    p = int(cycle["peak_frame"])
    e = int(cycle["end_frame"])

    seg = ang[s : e + 1]
    ext_seg = ang[s : p + 1]
    if len(ext_seg) < 3:
        ext_seg = seg

    # Derivatives over extension phase and full cycle
    ext_vel = np.gradient(ext_seg, dt) if len(ext_seg) >= 2 else np.zeros(1)
    ext_acc = np.gradient(ext_vel, dt) if len(ext_vel) >= 2 else np.zeros(1)
    ext_jerk = np.gradient(ext_acc, dt) if len(ext_acc) >= 2 else np.zeros(1)

    cyc_vel = np.gradient(seg, dt) if len(seg) >= 2 else np.zeros(1)
    cyc_acc = np.gradient(cyc_vel, dt) if len(cyc_vel) >= 2 else np.zeros(1)
    cyc_jerk = np.gradient(cyc_acc, dt) if len(cyc_acc) >= 2 else np.zeros(1)

    max_angle = float(np.max(seg))
    min_angle = float(np.min(seg))
    start_angle = float(ang[s])
    ext_rom = float(ang[p] - start_angle)
    total_rom = float(max_angle - min_angle)

    ext_dur = max(dt, float((p - s) * dt))
    cyc_dur = max(dt, float((e - s) * dt))

    # Log Dimensionless Jerk (LDLJ) - standard clinical rehabilitation smoothness metric
    v_peak = max(float(np.max(np.abs(ext_vel))), 1.0)
    jerk_integral = float(np.sum(ext_jerk ** 2) * dt)
    dimless_jerk = (jerk_integral * (ext_dur ** 3)) / (v_peak ** 2 + 1e-6)
    ext_ldlj = float(-np.log(max(dimless_jerk, 1e-6)))

    # Time / fraction near full anatomical extension
    near_peak_sec = float(np.sum(seg >= (max_angle - 4.0)) * dt)
    above_140_sec = float(np.sum(seg >= 140.0) * dt)
    above_145_sec = float(np.sum(seg >= 145.0) * dt)

    q25, q50, q75, q90, q95 = [float(x) for x in np.percentile(ext_seg, [25, 50, 75, 90, 95])]

    feats: dict[str, float] = {
        "max_angle": max_angle,
        "min_angle": min_angle,
        "start_angle": start_angle,
        "extension_rom": ext_rom,
        "total_rom": total_rom,
        "ext_mean_angle": float(np.mean(ext_seg)),
        "ext_std_angle": float(np.std(ext_seg)),
        "ext_q25_angle": q25,
        "ext_median_angle": q50,
        "ext_q75_angle": q75,
        "ext_q90_angle": q90,
        "ext_q95_angle": q95,
        "ext_iqr_angle": float(q75 - q25),
        "ext_duration_sec": ext_dur,
        "cycle_duration_sec": cyc_dur,
        "ext_vel_mean": float(np.mean(ext_vel)),
        "ext_vel_max": float(np.max(ext_vel)),
        "ext_vel_std": float(np.std(ext_vel)),
        "ext_acc_max": float(np.max(np.abs(ext_acc))),
        "ext_acc_std": float(np.std(ext_acc)),
        "ext_jerk_rms": float(np.sqrt(np.mean(ext_jerk ** 2))),
        "ext_ldlj": ext_ldlj,
        "cycle_vel_abs_mean": float(np.mean(np.abs(cyc_vel))),
        "cycle_jerk_rms": float(np.sqrt(np.mean(cyc_jerk ** 2))),
        "near_peak_sec": near_peak_sec,
        "above_140_sec": above_140_sec,
        "above_145_sec": above_145_sec,
        "ext_frac_above_135": float(np.mean(ext_seg >= 135.0)),
        "ext_frac_above_142": float(np.mean(ext_seg >= 142.0)),
        "ext_frac_above_148": float(np.mean(ext_seg >= 148.0)),
        "shoulder_disp_norm": float(kinematics.get("shoulder_disp_norm", 0.0)),
        "trunk_angle_std": float(kinematics.get("trunk_angle_std", 0.0)),
        "mean_visibility": float(kinematics.get("mean_visibility", 1.0)),
    }

    # Resample active extension trajectory [start_frame -> peak_frame] to T=16
    grid_ext_src = np.linspace(0.0, 1.0, len(ext_seg))
    grid_ext_dst = np.linspace(0.0, 1.0, ext_seq_len)
    ext_resampled = np.interp(grid_ext_dst, grid_ext_src, ext_seg)
    for k, val in enumerate(ext_resampled):
        feats[f"ext_traj_{k:02d}"] = float(val)

    # Resample full cycle [start_frame -> end_frame] to T=64 (for sequence analysis & visualization)
    grid_cyc_src = np.linspace(0.0, 1.0, len(seg))
    grid_cyc_dst = np.linspace(0.0, 1.0, seq_len)
    cyc_resampled = np.interp(grid_cyc_dst, grid_cyc_src, seg)
    for k, val in enumerate(cyc_resampled):
        feats[f"seq_{k:02d}"] = float(val)

    return feats
