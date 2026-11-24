"""Serializable calibrated model wrapper shared by training and frontend inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class CalibratedModelBundle:
    """Combines feature selection, StandardScaler, estimator, and probability calibration."""

    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        scaler: StandardScaler | None = None,
        calibrator: Any | None = None,
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.scaler = scaler
        self.calibrator = calibrator

    def _transform(self, X_df: pd.DataFrame) -> np.ndarray:
        X = X_df[self.feature_names].to_numpy(dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        X = self._transform(X_df)
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X)
        return self.model.predict_proba(X)

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X_df)[:, 1]
        return (probs >= 0.5).astype(int)
