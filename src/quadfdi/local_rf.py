"""Local 200 ms RF features and flight-cross-fitted probabilities.

The posterior is a classifier estimate, not a measured motor-health quantity.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .common import FitBatch, validate_windows, health_to_classes as physical_targets, class_flight_weights, write_json as _json
from .response import PREDICTOR_POINTS, context_points
EVIDENCE_SEED = 20261009
LOCAL_CUE_VERSION = 1
def physical_summary_features(windows):
    """Five blocks of ten physical features; no per-window normalization.

    Slopes are least-squares slopes per second over the genuine past points.
    Delta is last minus first; RMS includes the physical level.
    """
    x = validate_windows(windows).astype(np.float64)
    t = .02 * np.arange(x.shape[1], dtype=np.float64)
    centered = t - t.mean()
    slope = np.einsum("ntc,t->nc", x, centered) / np.dot(centered, centered)
    result = np.concatenate((x[:, -1], x.mean(1), x[:, -1] - x[:, 0],
                             np.sqrt(np.mean(x*x, axis=1)), slope), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Physical summary is nonfinite")
    return result.astype(np.float32)

def local_cue_features(windows):
    """A local 200 ms physical sequence plus its level/amplitude/trend summary."""
    x = validate_windows(windows)
    context_points(x.shape[1])
    local = x[:, -PREDICTOR_POINTS:]
    return np.concatenate((local.reshape(len(x), -1), physical_summary_features(local)), axis=1)

def _flight_folds(groups, seed, folds=3):
    ordered = sorted(np.unique(groups).tolist(), key=lambda g: hashlib.sha256(
        (str(seed)+"\0"+str(g)).encode()).hexdigest())
    if len(ordered) < folds:
        raise ValueError("At least three independent FIT flights are required")
    mapping = {str(g): i % folds for i, g in enumerate(ordered)}
    return np.asarray([mapping[str(g)] for g in groups], np.int64)

class LocalMotorCue:
    """OOF probabilities for fitting, all-FIT probabilities for evaluation."""
    def __init__(self, *, seed=EVIDENCE_SEED, folds=3, n_estimators=200, n_jobs=4):
        if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("Local cue seed must be a nonnegative integer")
        if folds != 3 or isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
            raise ValueError("Local cue requires three folds and positive integer trees")
        self.seed, self.folds = int(seed), int(folds)
        self.n_estimators, self.n_jobs = n_estimators, int(n_jobs)
        self.final_model = None
        self.fold_models = []
        self.audit = None

    def _model(self):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=self.n_estimators, min_samples_leaf=2,
                                      max_features="sqrt", random_state=self.seed,
                                      n_jobs=self.n_jobs, bootstrap=True)

    def fit(self, batch: FitBatch):
        import sklearn
        health, groups = batch.require_supervision()
        targets = physical_targets(health)
        features = local_cue_features(batch.windows)
        folds = _flight_folds(groups, self.seed, self.folds)
        self.fold_models, records = [], []
        for fold in range(self.folds):
            selected = folds != fold
            weights = class_flight_weights(targets[selected], groups[selected])
            model = self._model().fit(features[selected], targets[selected], sample_weight=weights)
            if not np.array_equal(model.classes_, np.arange(5)):
                raise ValueError("Each local cue training fold must contain all five classes")
            # Small online batches do not benefit from spawning a threaded
            # reduction. A fixed tree order also keeps cue probabilities exact
            # across reload and between optimization-seed runs.
            model.n_jobs = 1
            fitted = sorted(set(map(str, groups[selected])))
            held = sorted(set(map(str, groups[~selected])))
            if set(fitted) & set(held):
                raise ValueError("A held-out flight entered local cue fitting")
            records.append({"fold": fold, "classifier_fit_flights": fitted,
                            "cue_flights": held, "fit_windows": int(selected.sum()),
                            "fit_class_counts": np.bincount(targets[selected], minlength=5).tolist()})
            self.fold_models.append(model)
        weights = class_flight_weights(targets, groups)
        self.final_model = self._model().fit(features, targets, sample_weight=weights)
        self.final_model.n_jobs = 1
        self.history_points = context_points(batch.windows.shape[1])
        self.audit = {"version": LOCAL_CUE_VERSION, "seed": self.seed, "folds": records,
            "n_estimators": self.n_estimators, "n_jobs": self.n_jobs, "inference_n_jobs": 1,
            "sklearn_version": sklearn.__version__, "classes": ["healthy", "M1", "M2", "M3", "M4"],
            "history_points": self.history_points, "local_points": PREDICTOR_POINTS,
            "feature_count": int(features.shape[1]),
            "feature_rule": "last 11 physical points flattened + endpoint/mean/delta/RMS/least-squares slope",
            "fit_windows": len(features), "calibration_or_evaluation_fit_windows": 0,
            "fit_weight_rule": "healthy/fault 50/50; fault motors 12.5% each; equal flights within class",
            "fit_labels_used": True, "current_flight_labels_used_in_its_training_cue": False,
            "input_metadata_fields": [], "deployment_classifier_fit": "all supervised FIT windows",
            "tree_parameters": {"min_samples_leaf": 2, "max_features": "sqrt", "bootstrap": True}}
        return folds

    def transform(self, windows, *, fold=None):
        if self.final_model is None:
            raise ValueError("Fit or load the local cue first")
        x = validate_windows(windows)
        if x.shape[1] != self.history_points:
            raise ValueError("Local cue history differs from its FIT observations")
        if not len(x):
            return np.empty((0, 5), np.float64)
        if fold is not None and (isinstance(fold, bool) or not isinstance(fold, (int, np.integer))
                                 or not 0 <= fold < self.folds):
            raise ValueError("Local cue fold must be 0, 1 or 2")
        model = self.final_model if fold is None else self.fold_models[int(fold)]
        probability = model.predict_proba(local_cue_features(x))
        if (probability.shape != (len(x), 5) or not np.isfinite(probability).all()
                or np.any(probability < 0) or not np.allclose(probability.sum(1), 1)):
            raise ValueError("Local cue probabilities differ from the five-class contract")
        return probability

    def cross_fitted_transform(self, windows, fold_ids):
        x, ids = validate_windows(windows), np.asarray(fold_ids)
        if ids.shape != (len(x),) or ids.dtype.kind not in "iu" or np.any((ids < 0) | (ids >= self.folds)):
            raise ValueError("One valid held-out fold is required per training observation")
        result = np.empty((len(x), 5), np.float64)
        for fold in range(self.folds):
            selected = ids == fold
            result[selected] = self.transform(x[selected], fold=fold)
        return result

    def save(self, root):
        import joblib
        if self.final_model is None or self.audit is None:
            raise ValueError("Fit or load the local cue before saving")
        root = Path(root); root.mkdir(parents=True, exist_ok=True)
        path = root/"local_motor_cue.joblib"
        joblib.dump({"final": self.final_model, "folds": self.fold_models}, path, compress=3)
        _json(root/"local_motor_cue_audit.json", {**self.audit,
                                               "payload_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    @classmethod
    def load(cls, root):
        import joblib
        root = Path(root)
        audit = json.loads((root/"local_motor_cue_audit.json").read_text())
        path = root/"local_motor_cue.joblib"
        if (audit.get("version") != LOCAL_CUE_VERSION or audit.get("calibration_or_evaluation_fit_windows") != 0
                or audit.get("input_metadata_fields") != [] or len(audit.get("folds", [])) != 3
                or audit.get("local_points") != 11 or audit.get("feature_count") != 160
                or audit.get("classes") != ["healthy", "M1", "M2", "M3", "M4"]
                or audit.get("inference_n_jobs") != 1
                or audit.get("payload_sha256") != hashlib.sha256(path.read_bytes()).hexdigest()):
            raise ValueError("Local cue checkpoint contract differs")
        obj = cls(seed=audit["seed"], n_estimators=audit["n_estimators"], n_jobs=audit["n_jobs"])
        payload = joblib.load(path)
        obj.final_model, obj.fold_models = payload["final"], payload["folds"]
        obj.history_points, obj.audit = context_points(audit["history_points"]), audit
        if len(obj.fold_models) != obj.folds or any(not np.array_equal(model.classes_, np.arange(5))
                for model in [obj.final_model, *obj.fold_models]):
            raise ValueError("Local cue saved class order differs")
        if any(model.n_jobs != 1 or model.n_features_in_ != 160
               for model in [obj.final_model, *obj.fold_models]):
            raise ValueError("Local cue saved inference contract differs")
        for index, record in enumerate(audit["folds"]):
            if (record.get("fold") != index or not record.get("fit_windows", 0)
                    or not record.get("cue_flights") or not record.get("classifier_fit_flights")
                    or set(record["cue_flights"]) & set(record["classifier_fit_flights"])):
                raise ValueError("Local cue saved held-out flight contract differs")
        return obj
