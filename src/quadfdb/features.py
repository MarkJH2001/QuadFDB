"""The paper's 160 raw + 36 residual + 5 posterior representation."""
from __future__ import annotations

import hashlib
import json
import numpy as np

from .common import FitBatch, validate_windows
from .response import CausalEvidence
from .local_rf import LocalMotorCue

RAW_SCALE_FLOORS = np.asarray([1e-4] * 3 + [1e-3] * 3 + [1e-4] * 4)
MASKS = {"raw": (False, False), "residual": (True, False),
         "posterior": (False, True), "all": (True, True)}


def array_sha(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def mask_inputs(values, mask="all"):
    if mask not in MASKS:
        raise ValueError(f"Unknown input mask: {mask}")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 201 or not np.isfinite(values).all():
        raise ValueError("Expected finite 201-column feature bank")
    result = values.copy()
    residual, posterior = MASKS[mask]
    if not residual:
        result[:, 160:196] = 0
    if not posterior:
        result[:, 196:201] = 0
    return result


class EvidenceBank:
    """FIT-only feature tools; inference never uses event annotations.

    Response and RF models exclude the target flight for FIT feature generation.
    Raw scaling uses all healthy FIT observations. Residual centering/scaling
    uses pooled held-out healthy FIT errors, rather than separate fold scales.
    """

    def __init__(self, seed=20261009, threads=4):
        self.seed, self.threads = int(seed), int(threads)
        self.fitted = False

    @staticmethod
    def _windows(windows):
        x = validate_windows(windows)
        if x.shape[1:] != (16, 10):
            raise ValueError("The paper protocol uses 16 observations over 300 ms")
        return x

    def _join(self, windows, residuals, probabilities):
        n = len(windows)
        if residuals.shape != (n, 6, 6) or probabilities.shape != (n, 5):
            raise ValueError("Residual or posterior shape differs from the paper")
        raw = (windows.astype(np.float64) - self.raw_mean) / self.raw_scale
        values = np.concatenate((raw.reshape(n, 160), residuals.reshape(n, 36),
                                 probabilities), axis=1).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite representation")
        return values

    def fit_transform(self, batch: FitBatch):
        x = self._windows(batch.windows)
        health, groups = batch.require_supervision()
        healthy = x[batch.normal_rows()].astype(np.float64)
        self.raw_mean = healthy.mean(axis=(0, 1))
        self.raw_scale = np.maximum(healthy.std(axis=(0, 1)), RAW_SCALE_FLOORS)
        self.response = CausalEvidence(ridge=1e-4, folds=3, seed=self.seed,
                                       history_points=16)
        response_folds = self.response.fit(batch)
        residuals = self.response.cross_fitted_transform(x, response_folds)["residuals"]
        self.rf = LocalMotorCue(seed=self.seed, folds=3, n_estimators=200,
                                n_jobs=self.threads)
        rf_folds = self.rf.fit(batch)
        probabilities = self.rf.cross_fitted_transform(x, rf_folds)
        values = self._join(x, residuals, probabilities)
        self.audit = {
            "seed": self.seed, "feature_count": 201,
            "selected_fit_inputs_sha256": array_sha(x),
            "selected_fit_labels_sha256": array_sha(health),
            "selected_fit_groups_sha256": array_sha(groups),
            "OOF_residuals_sha256": array_sha(residuals),
            "OOF_RF_probabilities_sha256": array_sha(probabilities),
            "joined_201_inputs_sha256": array_sha(values),
            "raw_mean_sha256": array_sha(self.raw_mean),
            "raw_scale_sha256": array_sha(self.raw_scale),
            "evidence_fold_ids_sha256": array_sha(response_folds),
            "RF_fold_ids_sha256": array_sha(rf_folds),
            "CAL_or_TEST_used": False,
        }
        self.fitted = True
        return values

    def transform(self, windows):
        if not self.fitted:
            raise ValueError("Fit or load the feature bank first")
        x = self._windows(windows)
        return self._join(x, self.response.transform(x)["residuals"],
                          self.rf.transform(x))
