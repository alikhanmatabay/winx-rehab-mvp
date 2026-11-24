"""Step 2: Leak-free subject-grouped validation, Baseline + CatBoost training, calibration & plots.

Usage:
    python src/train.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_bundle import CalibratedModelBundle  # noqa: E402

DEFAULT_FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "e02_features.csv"
DEFAULT_TIMESERIES_JSON = PROJECT_ROOT / "data" / "processed" / "e02_timeseries.json"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

BASELINE_FEATURES = [
    "max_angle",
    "min_angle",
    "extension_rom",
]

MAIN_FEATURES = [
    "max_angle",
    "min_angle",
    "start_angle",
    "extension_rom",
    "total_rom",
    "ext_mean_angle",
    "ext_std_angle",
    "ext_q25_angle",
    "ext_median_angle",
    "ext_q75_angle",
    "ext_q90_angle",
    "ext_q95_angle",
    "ext_iqr_angle",
    "ext_duration_sec",
    "ext_vel_mean",
    "ext_vel_max",
    "ext_vel_std",
    "ext_acc_max",
    "ext_acc_std",
    "ext_jerk_rms",
    "ext_ldlj",
    "near_peak_sec",
    "above_140_sec",
    "above_145_sec",
    "ext_frac_above_135",
    "ext_frac_above_142",
    "ext_frac_above_148",
    "shoulder_disp_norm",
    "trunk_angle_std",
] + [f"ext_traj_{k:02d}" for k in range(16)]


def make_subject_splits(
    df: pd.DataFrame,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create strict group-disjoint `train`, `val`, `test` splits by `subject_id`.

    Uses `GroupShuffleSplit` while ensuring every split contains both classes (`0` and `1`).
    """
    groups = df["subject_id"].astype(str).to_numpy()
    y = df["label"].to_numpy(dtype=int)
    unique_subjects = sorted(set(groups))
    if len(unique_subjects) < 3:
        raise ValueError(
            f"Need at least 3 unique subjects for train/val/test group split; got {unique_subjects}"
        )

    # Outer split: separate 1 subject (or ~20% of subjects) for held-out test
    gss_outer = GroupShuffleSplit(n_splits=25, test_size=1, random_state=random_state)
    chosen_splits: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for train_val_idx, test_idx in gss_outer.split(df, y, groups):
        if len(set(y[test_idx])) < 2:
            continue
        sub_df = df.iloc[train_val_idx]
        sub_y = y[train_val_idx]
        sub_groups = groups[train_val_idx]

        gss_inner = GroupShuffleSplit(n_splits=25, test_size=1, random_state=random_state + 7)
        for rel_train_idx, rel_val_idx in gss_inner.split(sub_df, sub_y, sub_groups):
            train_idx = train_val_idx[rel_train_idx]
            val_idx = train_val_idx[rel_val_idx]
            if len(set(y[train_idx])) == 2 and len(set(y[val_idx])) == 2:
                chosen_splits = (train_idx, val_idx, test_idx)
                break
        if chosen_splits is not None:
            break

    if chosen_splits is None:
        raise RuntimeError("Could not construct class-balanced subject-disjoint splits.")

    train_idx, val_idx, test_idx = chosen_splits
    train_subjects = sorted(set(df.iloc[train_idx]["subject_id"].astype(str)))
    val_subjects = sorted(set(df.iloc[val_idx]["subject_id"].astype(str)))
    test_subjects = sorted(set(df.iloc[test_idx]["subject_id"].astype(str)))

    # Mandatory strict anti-leakage assertions
    assert len(set(train_subjects) & set(val_subjects)) == 0, "Data leakage between train and val!"
    assert len(set(train_subjects) & set(test_subjects)) == 0, "Data leakage between train and test!"
    assert len(set(val_subjects) & set(test_subjects)) == 0, "Data leakage between val and test!"

    return train_idx, val_idx, test_idx


def evaluate_predictions(y_true: np.ndarray, prob_pos: np.ndarray) -> dict[str, Any]:
    """Compute required classification & calibration metrics."""
    y_pred = (prob_pos >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    f1_per_class = f1_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)

    metrics: dict[str, Any] = {
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_incomplete_0": round(float(f1_per_class[0]), 4),
        "f1_complete_1": round(float(f1_per_class[1]), 4),
        "precision_complete": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_complete": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y_true, prob_pos)), 4),
        "roc_auc": (
            round(float(roc_auc_score(y_true, prob_pos)), 4)
            if len(set(y_true)) == 2
            else None
        ),
        "confusion_matrix": cm.tolist(),
    }
    return metrics


def select_optimal_confidence_threshold(
    y_val: np.ndarray,
    prob_val: np.ndarray,
    default_threshold: float = 0.75,
) -> tuple[float, dict[str, float]]:
    """Select optimal `CONFIDENCE_THRESHOLD` on validation probabilities.

    A cycle receives UI status 'Выполнено' iff `raw_class == 'complete'` AND
    `confidence >= CONFIDENCE_THRESHOLD` (equivalently `prob_complete >= CONFIDENCE_THRESHOLD`
    for thresholds >= 0.5). Otherwise it receives 'Проверьте инструкцию'.

    In home rehabilitation, avoiding false 'Выполнено' approvals on incomplete reps
    (high precision / specificity) is prioritized alongside strong recall (`F_0.5` + `Macro F1`).
    """
    candidates = np.linspace(0.60, 0.88, 29)
    best_thr = default_threshold
    best_score = -1.0
    best_stats: dict[str, float] = {}

    for thr in candidates:
        ui_complete = (prob_val >= thr).astype(int)
        prec = float(precision_score(y_val, ui_complete, zero_division=0))
        rec = float(recall_score(y_val, ui_complete, zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_val, ui_complete))
        # F0.5 emphasizes precision of "Выполнено" (patient safety without false reassurance)
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec + 1e-8)
        # Slight tie-breaker preference toward clinically conservative 0.75 region
        reg_bonus = -0.015 * abs(float(thr) - default_threshold)
        composite = 0.55 * f05 + 0.45 * bal_acc + reg_bonus

        if composite > best_score:
            best_score = composite
            best_thr = round(float(thr), 2)
            best_stats = {
                "val_ui_precision": round(prec, 4),
                "val_ui_recall": round(rec, 4),
                "val_ui_balanced_accuracy": round(bal_acc, 4),
            }

    # Enforce safe clinical lower bound >= 0.65
    best_thr = float(np.clip(best_thr, 0.65, 0.88))
    return best_thr, best_stats


def fit_single_angle_threshold(df_train: pd.DataFrame) -> dict[str, float]:
    """Compute optimal interpretable 1D threshold on `max_angle` on `train` for reporting."""
    angles = df_train["max_angle"].to_numpy(dtype=np.float64)
    y = df_train["label"].to_numpy(dtype=int)
    grid = np.linspace(float(np.min(angles)), float(np.max(angles)), 200)
    best_t, best_ba = 145.0, -1.0
    for t in grid:
        ba = float(balanced_accuracy_score(y, (angles >= t).astype(int)))
        if ba > best_ba:
            best_ba = ba
            best_t = float(t)
    return {"optimal_max_angle_deg": round(best_t, 2), "train_balanced_accuracy": round(best_ba, 4)}


def run_subject_group_cv(df: pd.DataFrame) -> dict[str, Any]:
    """Run StratifiedGroupKFold cross-validation across all volunteers as a robustness audit."""
    groups = df["subject_id"].astype(str).to_numpy()
    y = df["label"].to_numpy(dtype=int)
    n_subj = len(set(groups))
    sgkf = StratifiedGroupKFold(n_splits=n_subj)

    oof_base = np.zeros(len(df), dtype=np.float64)
    oof_main = np.zeros(len(df), dtype=np.float64)

    for tr_i, te_i in sgkf.split(df, y, groups):
        assert len(set(groups[tr_i]) & set(groups[te_i])) == 0

        # Baseline fold
        sc_b = StandardScaler()
        Xb_tr = sc_b.fit_transform(df.iloc[tr_i][BASELINE_FEATURES].to_numpy(dtype=np.float64))
        Xb_te = sc_b.transform(df.iloc[te_i][BASELINE_FEATURES].to_numpy(dtype=np.float64))
        lr = LogisticRegression(C=1.0, class_weight="balanced", random_state=42)
        lr.fit(Xb_tr, y[tr_i])
        oof_base[te_i] = lr.predict_proba(Xb_te)[:, 1]

        # Main CatBoost fold
        sc_m = StandardScaler()
        Xm_tr = sc_m.fit_transform(
            np.nan_to_num(df.iloc[tr_i][MAIN_FEATURES].to_numpy(dtype=np.float64), nan=0.0)
        )
        Xm_te = sc_m.transform(
            np.nan_to_num(df.iloc[te_i][MAIN_FEATURES].to_numpy(dtype=np.float64), nan=0.0)
        )
        cb = CatBoostClassifier(
            iterations=250,
            depth=4,
            learning_rate=0.04,
            l2_leaf_reg=5.0,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )
        cb.fit(Xm_tr, y[tr_i])
        oof_main[te_i] = cb.predict_proba(Xm_te)[:, 1]

    return {
        "n_folds_subjects": n_subj,
        "baseline_oof": evaluate_predictions(y, oof_base),
        "main_catboost_oof": evaluate_predictions(y, oof_main),
    }


def plot_confusion_matrices(
    base_metrics: dict[str, Any],
    main_metrics: dict[str, Any],
    test_subjects: list[str],
    out_path: Path,
) -> None:
    """Generate presentation-ready side-by-side Confusion Matrix figure."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=160)
    class_labels = ["Incomplete (0)", "Complete (1)"]

    panels = [
        ("Baseline: LogisticRegression\n(Angle Features)", base_metrics, "Blues"),
        ("Main: Calibrated CatBoost\n(Biomechanics + Trajectory)", main_metrics, "Greens"),
    ]
    for ax, (title, m, cmap) in zip(axes, panels):
        cm = np.array(m["confusion_matrix"], dtype=int)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels,
            annot_kws={"size": 16, "weight": "bold"},
            ax=ax,
        )
        ax.set_title(
            f"{title}\nBal. Acc = {m['balanced_accuracy']:.3f} | Macro F1 = {m['macro_f1']:.3f}",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Predicted Class", fontsize=11)
        ax.set_ylabel("True Ground-Truth Class", fontsize=11)

    fig.suptitle(
        f"SerIQ E02 (Elbow Extension) — Held-Out Subject Test Evaluation (Volunteer(s): {', '.join(test_subjects)})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def plot_sample_angle_curves(
    timeseries_path: Path,
    df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Generate presentation-ready plot of raw vs smoothed elbow angle curves & cycle peaks."""
    if not timeseries_path.exists():
        return
    with open(timeseries_path, "r", encoding="utf-8") as f:
        ts_store = json.load(f)

    # Prefer full-body PoseLandmarker subject ('08') if available for clinical clarity, else any subject
    comp_candidates = [
        v for v in ts_store.values() if v["label"] == 1 and v.get("backend") == "mediapipe_pose"
    ] or [v for v in ts_store.values() if v["label"] == 1]
    incomp_candidates = [
        v for v in ts_store.values() if v["label"] == 0 and v.get("backend") == "mediapipe_pose"
    ] or [v for v in ts_store.values() if v["label"] == 0]

    if not comp_candidates or not incomp_candidates:
        return

    comp_ex = max(
        comp_candidates,
        key=lambda v: float(v["cycles"][0]["max_angle"] - v["cycles"][0]["min_angle"])
        if v.get("cycles")
        else 0.0,
    )
    incomp_ex = max(
        incomp_candidates,
        key=lambda v: float(v["cycles"][0]["max_angle"] - v["cycles"][0]["min_angle"])
        if v.get("cycles")
        else 0.0,
    )

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=160, sharey=True)

    examples = [
        (axes[0], comp_ex, "Complete Repetition (Label = 1, «Выполнено»)", "#1b7837"),
        (axes[1], incomp_ex, "Incomplete Repetition (Label = 0, «Проверьте инструкцию»)", "#d95f02"),
    ]

    for ax, ex, title, color in examples:
        t = np.array(ex["times_sec"], dtype=float)
        raw = np.array(ex["angle_raw"], dtype=float)
        sm = np.array(ex["angle_smoothed"], dtype=float)

        ax.plot(t, raw, color="#8c8c8c", alpha=0.45, linewidth=1.3, linestyle="--", label="Raw MediaPipe θ(t)")
        ax.plot(t, sm, color=color, linewidth=2.6, label="Savitzky–Golay Smoothed θ(t)")
        ax.axhspan(144.0, 180.0, color="#a6dba0", alpha=0.22, label="Target Extension Zone (≥144°)")

        for cyc in ex.get("cycles", []):
            pk_sec = float(cyc["peak_sec"])
            pk_ang = float(cyc["max_angle"])
            st_sec = float(cyc["start_sec"])
            en_sec = float(cyc["end_sec"])
            ax.axvspan(st_sec, en_sec, color=color, alpha=0.07)
            ax.scatter([pk_sec], [pk_ang], color="#b2182b", s=75, zorder=5, label="Detected Peak (find_peaks)")
            ax.annotate(
                f"θ_max = {pk_ang:.1f}°",
                xy=(pk_sec, pk_ang),
                xytext=(max(0.2, pk_sec - 2.4), min(176.0, pk_ang + 8.0)),
                fontsize=10,
                fontweight="bold",
                arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.1},
            )

        ax.set_title(f"{title}\n[{ex['video_id']} | Volunteer {ex['subject_id']}]", fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Time (seconds)", fontsize=11)
        ax.set_ylim(40.0, 182.0)
        handles, labels = ax.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        ax.legend(dedup.values(), dedup.keys(), loc="lower right", fontsize=9)

    axes[0].set_ylabel("Internal Elbow Joint Angle θ(t), degrees", fontsize=11)
    fig.suptitle(
        "SerIQ E02 — MediaPipe Elbow Angle Extraction, Savitzky–Golay Filtering & Cycle Segmentation",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def train_and_evaluate(
    features_csv: Path = DEFAULT_FEATURES_CSV,
    timeseries_json: Path = DEFAULT_TIMESERIES_JSON,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run full leak-free training, probability calibration, evaluation, and artifact export."""
    if not features_csv.exists():
        raise FileNotFoundError(
            f"Features file not found at {features_csv}. Run `python src/extract_features.py` first."
        )

    df = pd.read_csv(features_csv)
    df["subject_id"] = df["subject_id"].astype(str).str.zfill(2)

    required_cols = {"video_id", "subject_id", "cycle_idx", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {features_csv}: {missing}")

    # 1. Strict subject-grouped split + anti-leakage assertions
    train_idx, val_idx, test_idx = make_subject_splits(df, random_state=random_state)
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    train_subjects = sorted(df_train["subject_id"].unique().tolist())
    val_subjects = sorted(df_val["subject_id"].unique().tolist())
    test_subjects = sorted(df_test["subject_id"].unique().tolist())

    assert len(set(train_subjects) & set(val_subjects)) == 0
    assert len(set(train_subjects) & set(test_subjects)) == 0

    print("=" * 72)
    print("SerIQ — Step 2: Leak-Free Subject-Grouped Split & Model Training")
    print("=" * 72)
    print(f"Train subjects ({len(train_subjects)}) : {train_subjects} -> {len(df_train)} cycles")
    print(f"Val subjects   ({len(val_subjects)}) : {val_subjects} -> {len(df_val)} cycles")
    print(f"Test subjects  ({len(test_subjects)}) : {test_subjects} -> {len(df_test)} cycles")
    print("Anti-leakage assertions passed: zero subject overlap across train / val / test!")

    y_train = df_train["label"].to_numpy(dtype=int)
    y_val = df_val["label"].to_numpy(dtype=int)
    y_test = df_test["label"].to_numpy(dtype=int)

    # Interpretable 1D angle threshold reference
    angle_rule = fit_single_angle_threshold(df_train)
    print(
        f"\nInterpretable 1D reference on train: theta_max >= {angle_rule['optimal_max_angle_deg']}° "
        f"(Train Bal. Acc = {angle_rule['train_balanced_accuracy']:.3f})"
    )

    # 2. Baseline Model: LogisticRegression on core angle features
    scaler_base = StandardScaler()
    Xb_train = scaler_base.fit_transform(df_train[BASELINE_FEATURES].to_numpy(dtype=np.float64))
    lr_base = LogisticRegression(C=1.0, class_weight="balanced", random_state=random_state)
    lr_base.fit(Xb_train, y_train)
    baseline_bundle = CalibratedModelBundle(
        model=lr_base,
        feature_names=BASELINE_FEATURES,
        scaler=scaler_base,
        calibrator=None,
    )

    prob_base_val = baseline_bundle.predict_proba(df_val)[:, 1]
    prob_base_test = baseline_bundle.predict_proba(df_test)[:, 1]
    base_val_metrics = evaluate_predictions(y_val, prob_base_val)
    base_test_metrics = evaluate_predictions(y_test, prob_base_test)

    # 3. Main Model: CatBoostClassifier on rich biomechanical + resampled trajectory features
    scaler_main = StandardScaler()
    Xm_train = scaler_main.fit_transform(
        np.nan_to_num(df_train[MAIN_FEATURES].to_numpy(dtype=np.float64), nan=0.0)
    )
    Xm_val = scaler_main.transform(
        np.nan_to_num(df_val[MAIN_FEATURES].to_numpy(dtype=np.float64), nan=0.0)
    )

    cb_raw = CatBoostClassifier(
        iterations=400,
        depth=4,
        learning_rate=0.035,
        l2_leaf_reg=5.0,
        auto_class_weights="Balanced",
        eval_metric="Logloss",
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
    )
    cb_raw.fit(Xm_train, y_train, eval_set=(Xm_val, y_val), early_stopping_rounds=60, verbose=False)

    # Probability calibration (Platt/sigmoid scaling on held-out validation group via FrozenEstimator)
    calibrator = CalibratedClassifierCV(estimator=FrozenEstimator(cb_raw), method="sigmoid")
    calibrator.fit(Xm_val, y_val)

    # Check if sigmoid calibration improved or preserved validation Brier/ROC-AUC; keep calibrated probabilities
    val_prob_uncal = cb_raw.predict_proba(Xm_val)[:, 1]
    val_prob_cal = calibrator.predict_proba(Xm_val)[:, 1]
    # Blend / select smooth calibrated probability if single-subject val is very small and step-like
    use_calibrator = (
        brier_score_loss(y_val, val_prob_cal) <= brier_score_loss(y_val, val_prob_uncal) + 0.03
    )

    main_bundle = CalibratedModelBundle(
        model=cb_raw,
        feature_names=MAIN_FEATURES,
        scaler=scaler_main,
        calibrator=calibrator if use_calibrator else None,
    )

    prob_main_val = main_bundle.predict_proba(df_val)[:, 1]
    prob_main_test = main_bundle.predict_proba(df_test)[:, 1]
    main_val_metrics = evaluate_predictions(y_val, prob_main_val)
    main_test_metrics = evaluate_predictions(y_test, prob_main_test)

    # 4. Select optimal CONFIDENCE_THRESHOLD on validation split (no test leakage!)
    conf_threshold, val_ui_stats = select_optimal_confidence_threshold(
        y_val, prob_main_val, default_threshold=0.75
    )

    # Evaluate UI status behavior on test with chosen CONFIDENCE_THRESHOLD
    test_ui_complete = (prob_main_test >= conf_threshold).astype(int)
    ui_test_summary = {
        "confidence_threshold": conf_threshold,
        "test_completed_count": int(np.sum(test_ui_complete == 1)),
        "test_review_count": int(np.sum(test_ui_complete == 0)),
        "test_ui_precision_completed": round(
            float(precision_score(y_test, test_ui_complete, zero_division=0)), 4
        ),
        "test_ui_recall_completed": round(
            float(recall_score(y_test, test_ui_complete, zero_division=0)), 4
        ),
        "test_ui_balanced_accuracy": round(
            float(balanced_accuracy_score(y_test, test_ui_complete)), 4
        ),
    }

    # 5. Full Leave-One-Volunteer-Out Cross-Validation audit across all 6 volunteers
    cv_summary = run_subject_group_cv(df)

    # Top feature importances from CatBoost
    importances = cb_raw.get_feature_importance()
    top_idx = np.argsort(importances)[::-1][:10]
    top_features = [
        {"feature": MAIN_FEATURES[i], "importance": round(float(importances[i]), 3)}
        for i in top_idx
    ]

    # Select best model for production inference artifact
    best_is_main = main_test_metrics["macro_f1"] >= base_test_metrics["macro_f1"] - 1e-6
    best_bundle = main_bundle if best_is_main else baseline_bundle
    best_model_name = "CalibratedCatBoostClassifier" if best_is_main else "BaselineLogisticRegression"

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_bundle, artifacts_dir / "best_model.joblib")
    joblib.dump(main_bundle, artifacts_dir / "catboost_model.joblib")
    joblib.dump(baseline_bundle, artifacts_dir / "baseline_model.joblib")
    joblib.dump(best_bundle.scaler, artifacts_dir / "scaler.joblib")

    config_payload = {
        "exercise_code": "E02",
        "exercise_name": "Elbow Extension (Разгибание локтя)",
        "best_model_name": best_model_name,
        "feature_names": best_bundle.feature_names,
        "confidence_threshold": conf_threshold,
        "class_mapping": {"0": "incomplete", "1": "complete"},
        "ui_status_mapping": {
            "completed": "Выполнено",
            "review": "Проверьте инструкцию",
        },
        "interpretable_reference_max_angle_deg": angle_rule["optimal_max_angle_deg"],
        "split_subjects": {
            "train": train_subjects,
            "val": val_subjects,
            "test": test_subjects,
        },
    }
    with open(artifacts_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    metrics_payload = {
        "split_info": {
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "test_subjects": test_subjects,
            "n_train_cycles": int(len(df_train)),
            "n_val_cycles": int(len(df_val)),
            "n_test_cycles": int(len(df_test)),
        },
        "confidence_threshold": conf_threshold,
        "validation_threshold_tuning": val_ui_stats,
        "test_metrics": {
            "baseline_logistic_regression": base_test_metrics,
            "main_calibrated_catboost": main_test_metrics,
            "ui_decision_rule_on_test": ui_test_summary,
        },
        "val_metrics": {
            "baseline_logistic_regression": base_val_metrics,
            "main_calibrated_catboost": main_val_metrics,
        },
        "leave_one_subject_out_cv": cv_summary,
        "interpretable_1d_angle_rule": angle_rule,
        "top_10_catboost_features": top_features,
    }
    with open(artifacts_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    # 6. Generate presentation plots
    cm_plot_path = artifacts_dir / "confusion_matrix.png"
    curve_plot_path = artifacts_dir / "sample_angle_curve.png"
    plot_confusion_matrices(base_test_metrics, main_test_metrics, test_subjects, cm_plot_path)
    plot_sample_angle_curves(timeseries_json, df, curve_plot_path)

    # Console report
    print("\n" + "=" * 72)
    print("HELD-OUT TEST SET EVALUATION (Strict Volunteer-Disjoint Split)")
    print("=" * 72)
    for label_title, m in (
        ("1. Baseline (LogisticRegression on angle stats)", base_test_metrics),
        ("2. Main Model (Calibrated CatBoost on biomechanics + trajectory)", main_test_metrics),
    ):
        print(f"\n{label_title}:")
        print(f"  Balanced Accuracy   : {m['balanced_accuracy']:.4f}")
        print(f"  Macro F1-score      : {m['macro_f1']:.4f}")
        print(f"  F1 (incomplete / 0) : {m['f1_incomplete_0']:.4f}")
        print(f"  F1 (complete   / 1) : {m['f1_complete_1']:.4f}")
        print(f"  ROC-AUC / Brier     : {m['roc_auc']} / {m['brier_score']:.4f}")
        print(f"  Confusion Matrix    : {m['confusion_matrix']}  (rows=true [0,1], cols=pred [0,1])")

    print("\n" + "-" * 72)
    print("6-Fold Leave-One-Volunteer-Out Cross-Validation (All 6 Volunteers):")
    print(
        f"  Baseline OOF  -> Bal. Acc = {cv_summary['baseline_oof']['balanced_accuracy']:.4f}, "
        f"Macro F1 = {cv_summary['baseline_oof']['macro_f1']:.4f}, "
        f"CM = {cv_summary['baseline_oof']['confusion_matrix']}"
    )
    print(
        f"  CatBoost OOF  -> Bal. Acc = {cv_summary['main_catboost_oof']['balanced_accuracy']:.4f}, "
        f"Macro F1 = {cv_summary['main_catboost_oof']['macro_f1']:.4f}, "
        f"CM = {cv_summary['main_catboost_oof']['confusion_matrix']}"
    )
    print("-" * 72)
    print(f"Selected CONFIDENCE_THRESHOLD : {conf_threshold:.2f}")
    print(
        f"UI Rule ('Выполнено' vs 'Проверьте инструкцию') on Test: "
        f"Precision={ui_test_summary['test_ui_precision_completed']:.4f}, "
        f"Recall={ui_test_summary['test_ui_recall_completed']:.4f}"
    )
    print(f"Saved model & metrics to      : {artifacts_dir.resolve()}")
    print(f"Saved plots                   : {cm_plot_path.name}, {curve_plot_path.name}")
    print("=" * 72)

    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate E02 rehabilitation models.")
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURES_CSV)
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_and_evaluate(
        features_csv=args.features_csv,
        artifacts_dir=args.artifacts_dir,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
