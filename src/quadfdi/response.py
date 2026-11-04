"""Healthy ridge prediction and flight-cross-fitted causal residuals.

RAGB uses only the signed ``residuals`` output. Other shared numerical helper
outputs are retained for source compatibility and do not enter its 201 inputs.
All calculations use observations at or before the scored endpoint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .common import FitBatch, validate_windows
from .common import ridge_operator

CONTEXT_POINTS = 64
PREDICTOR_POINTS = 11
VERSION = 1
SUPPORTED_CONTEXT_POINTS = (11, 16, 64)


def context_points(value):
    if type(value) is not int or value not in SUPPORTED_CONTEXT_POINTS:
        raise ValueError("Evidence history_points must be 11, 16 or 64 real observations")
    return value


def response_design(windows, mean, scale):
    x = (np.asarray(windows, dtype=np.float64) - mean) / scale
    return np.concatenate((x[:, :-1, :6].reshape(len(x), -1),
                           x[:, :-1, 6:].reshape(len(x), -1), np.ones((len(x), 1))), axis=1)


class ResponseModel:
    def fit(self, windows, ridge):
        x = validate_windows(windows)
        if x.shape[1] != PREDICTOR_POINTS or not len(x):
            raise ValueError("Healthy response fit requires nonempty 11-point windows")
        self.mean = x.mean(axis=(0, 1), dtype=np.float64)
        self.scale = np.maximum(x.std(axis=(0, 1), dtype=np.float64),
                                [1e-4]*3 + [1e-3]*3 + [1e-4]*4)
        self.operator = ridge_operator(response_design(x, self.mean, self.scale),
                                       (x[:, -1, :6] - self.mean[:6]) / self.scale[:6], ridge)
        return self

    def predict(self, windows):
        return response_design(windows, self.mean, self.scale) @ self.operator * self.scale[:6] + self.mean[:6]

    def input_effects(self, windows):
        # Remove a fraction of the *physical* command, then standardize. Scaling
        # a centered command directly would incorrectly scale its offset.
        u = np.asarray(windows, np.float64)[:, :-1, 6:] / self.scale[6:]
        b = self.operator[60:100].reshape(10, 4, 6) * self.scale[:6]
        return np.einsum("ntm,tmc->nmc", u, b)


def hypothesis_evidence(errors, effects, inverse_covariance):
    e, v, p = map(lambda x: np.asarray(x, np.float64), (errors, effects, inverse_covariance))
    baseline = np.einsum("nc,cd,nd->n", e, p, e)
    leverage2 = np.maximum(np.einsum("nmc,cd,nmd->nm", v, p, v), 0.)
    cross = np.einsum("nmc,cd,nd->nm", v, p, e)
    reduction = np.clip(-cross / np.maximum(leverage2, 1e-8), 0., 1.)
    reduction[leverage2 < 1e-8] = 0.
    improvement = np.clip((-(2*reduction*cross + reduction**2*leverage2)) /
                          np.maximum(baseline[:, None], 1e-6), 0., 1.)
    return reduction, improvement, np.sqrt(leverage2)


class CausalEvidence:
    """Three flight folds for FIT evidence; all healthy FIT for deployment."""
    def __init__(self, *, ridge=1e-4, folds=3, seed=20261009, history_points=CONTEXT_POINTS):
        self.ridge, self.folds, self.seed = float(ridge), int(folds), int(seed)
        self.history_points = context_points(history_points)
        if self.ridge <= 0 or self.folds != 3:
            raise ValueError("The response feature contract fixes positive ridge and three flight folds")

    def fit(self, batch: FitBatch):
        _, groups = batch.require_supervision()
        x, normal = batch.windows, batch.normal_rows()
        if x.shape[1] != self.history_points:
            raise ValueError(f"QuadFDI requires {self.history_points} observed points")
        ordered = sorted(np.unique(groups).tolist(), key=lambda g: hashlib.sha256(
            (str(self.seed)+"\0"+str(g)).encode()).hexdigest())
        if len(ordered) < self.folds:
            raise ValueError("At least three independent fit flights are required")
        assignments = {str(g): i % self.folds for i, g in enumerate(ordered)}
        fold_ids = np.asarray([assignments[str(g)] for g in groups])
        self.fold_models, records = [], []
        endpoint_errors = np.empty((int(normal.sum()), 6), np.float64)
        normal_indices = np.flatnonzero(normal)
        for fold in range(self.folds):
            selected = normal & (fold_ids != fold)
            if not selected.any() or not (normal & (fold_ids == fold)).any():
                raise ValueError("Every flight fold requires held-out and fitted healthy observations")
            model = ResponseModel().fit(x[selected, -PREDICTOR_POINTS:], self.ridge)
            self.fold_models.append(model)
            held = fold_ids[normal_indices] == fold
            query = x[normal_indices[held], -PREDICTOR_POINTS:]
            endpoint_errors[held] = query[:, -1, :6] - model.predict(query)
            fit_groups = sorted(set(map(str, groups[selected])))
            held_groups = sorted(set(map(str, groups[fold_ids == fold])))
            if set(fit_groups) & set(held_groups):
                raise ValueError("A held-out flight entered response fitting")
            records.append({"fold": fold, "predictor_fit_flights": fit_groups,
                            "evidence_flights": held_groups, "healthy_fit_windows": int(selected.sum())})
        self.error_mean = endpoint_errors.mean(axis=0)
        covariance = np.cov(endpoint_errors, rowvar=False)
        diagonal = np.maximum(np.diag(covariance), np.array([1e-5]*3+[1e-4]*3)**2)
        self.covariance = .9*covariance + .1*np.diag(diagonal) + np.diag(diagonal*1e-6)
        self.inverse_covariance = np.linalg.inv(self.covariance)
        self.error_scale = np.sqrt(np.diag(self.covariance))
        self.final_model = ResponseModel().fit(x[normal, -PREDICTOR_POINTS:], self.ridge)
        self.audit = {"version": VERSION, "seed": self.seed, "ridge": self.ridge,
            "folds": records, "healthy_fit_windows": int(normal.sum()),
            "calibration_or_evaluation_fit_windows": 0, "endpoint_in_predictor_input": False,
            "residual_scale_fit": "held-out healthy FIT endpoints only",
            "deployment_predictor_fit": "all healthy FIT windows only",
            "hypothesis": "past command attenuation in a fitted linear response; soft evidence",
            "residual_points": self.history_points-PREDICTOR_POINTS+1}
        if self.history_points != CONTEXT_POINTS:
            self.audit.update(version=2, history_points=self.history_points,
                predictor_points=PREDICTOR_POINTS, sample_interval_ms=20,
                short_residual_points=min(self.history_points-PREDICTOR_POINTS+1,11),
                residual_trend="endpoint difference divided by actual observed residual span; zero for a single residual")
        return fold_ids

    def transform(self, windows, *, fold=None):
        x = validate_windows(windows)
        if x.shape[1] != self.history_points:
            raise ValueError(f"Expected {self.history_points}-point observation windows")
        n, h = len(x), self.history_points-PREDICTOR_POINTS+1
        if not n:
            return {"residuals": np.empty((0,h,6),np.float32), "features": np.empty((0,45),np.float32)}
        parts = np.lib.stride_tricks.sliding_window_view(x, PREDICTOR_POINTS, axis=1).transpose(0,1,3,2)
        flat = parts.reshape(-1, PREDICTOR_POINTS, 10)
        model = self.final_model if fold is None else self.fold_models[fold]
        errors = flat[:, -1, :6] - model.predict(flat) - self.error_mean
        z = (errors / self.error_scale).reshape(n,h,6)
        norm = np.sqrt(np.maximum(np.einsum("nc,cd,nd->n", errors, self.inverse_covariance, errors),0.)).reshape(n,h)
        tail = z[:, -11:]
        # Eleven observed points produce exactly one prediction error. There is
        # no measured residual trend in that case, and no extra past is added.
        trend = ((tail[:, -1]-tail[:, 0]) / (.02*(tail.shape[1]-1))
                 if tail.shape[1] > 1 else np.zeros((n,6),np.float64))
        reduction, improvement, leverage = hypothesis_evidence(
            errors.reshape(n,h,6)[:, -1], model.input_effects(x[:, -11:]), self.inverse_covariance)
        features = np.concatenate((z[:, -1], tail.mean(1), np.sqrt((tail**2).mean(1)),
            np.abs(tail).max(1), trend,
            norm[:, -1, None], norm[:, -11:].mean(1)[:,None], norm[:, -11:].max(1)[:,None],
            reduction, improvement, np.log1p(leverage)), axis=1).astype(np.float32)
        if not np.isfinite(z).all() or not np.isfinite(features).all():
            raise ValueError("Numerical evidence is nonfinite")
        return {"residuals": z.astype(np.float32), "features": features}

    def cross_fitted_transform(self, windows, fold_ids):
        rows = self.transform(np.empty((0,self.history_points,10), np.float32))
        rows = {k: np.empty((len(windows), *v.shape[1:]),np.float32) for k,v in rows.items()}
        for fold in range(self.folds):
            indices = np.flatnonzero(fold_ids == fold)
            for start in range(0,len(indices),256):
                chosen = indices[start:start+256]
                transformed = self.transform(windows[chosen], fold=fold)
                for key in rows:
                    rows[key][chosen] = transformed[key]
        return rows

    def save(self, root):
        root = Path(root)
        arrays = {"error_mean": self.error_mean, "covariance": self.covariance}
        for prefix, model in [("final",self.final_model)] + [("fold"+str(i),m) for i,m in enumerate(self.fold_models)]:
            for name in ("mean","scale","operator"):
                arrays[prefix+"_"+name] = getattr(model,name)
        np.savez_compressed(root/"response_evidence.npz", **arrays)
        (root/"evidence_audit.json").write_text(json.dumps(self.audit,indent=2)+"\n")

    @classmethod
    def load(cls, root):
        root = Path(root)
        audit = json.loads((root/"evidence_audit.json").read_text())
        if audit.get("version") not in (VERSION,2) or audit.get("calibration_or_evaluation_fit_windows") != 0:
            raise ValueError("Response evidence contract differs")
        obj = cls(ridge=audit["ridge"], seed=audit["seed"],history_points=audit.get("history_points",CONTEXT_POINTS))
        if audit.get("residual_points") != obj.history_points-PREDICTOR_POINTS+1:
            raise ValueError("Response evidence observed history differs")
        if audit["version"] != (VERSION if obj.history_points==CONTEXT_POINTS else 2):
            raise ValueError("Response evidence history version differs")
        if audit["version"]==2 and (audit.get("predictor_points")!=PREDICTOR_POINTS or
                audit.get("sample_interval_ms")!=20 or audit.get("short_residual_points")!=obj.history_points-10):
            raise ValueError("Response evidence short history contract differs")
        with np.load(root/"response_evidence.npz",allow_pickle=False) as arrays:
            obj.error_mean, obj.covariance = arrays["error_mean"], arrays["covariance"]
            if obj.error_mean.shape != (6,) or obj.covariance.shape != (6,6):
                raise ValueError("Evidence covariance shape differs")
            obj.error_scale = np.sqrt(np.diag(obj.covariance))
            obj.inverse_covariance = np.linalg.inv(obj.covariance)
            models = []
            for prefix in ("final","fold0","fold1","fold2"):
                model = ResponseModel()
                for field, shape in (("mean",(10,)),("scale",(10,)),("operator",(101,6))):
                    value = arrays[prefix+"_"+field]
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError("Evidence predictor shape or values differ")
                    setattr(model,field,value)
                if np.any(model.scale <= 0):
                    raise ValueError("Evidence standardizer must have positive scales")
                models.append(model)
        obj.final_model, obj.fold_models, obj.audit = models[0], models[1:], audit
        return obj
