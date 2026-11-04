"""Flattened-history tabular references; neural primitives are in neural.py."""
from __future__ import annotations
import numpy as np
from .common import (FitBatch, ScoreBatch, health_to_classes,
                     binary_flight_weights, validate_windows)
from .models import HGB_PARAMETERS


class TabularReference:
    """Healthy-FIT scaling followed by fixed ET, SVM or raw HGB.

    The separate RFReference uses physical local histories and summaries;
    it is not a flattened 300 ms forest.
    """
    def __init__(self, method="hgb", seed=20261009, threads=4):
        if method not in ("et", "svm", "hgb"):
            raise ValueError("Use et, svm or hgb")
        self.method, self.seed, self.threads = method, int(seed), int(threads)

    def fit(self, batch: FitBatch):
        from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
        from sklearn.svm import SVC
        from threadpoolctl import threadpool_limits
        health, groups = batch.require_supervision()
        x = batch.windows
        if x.shape[1:] != (16, 10):
            raise ValueError("Expected the paper's 16 by 10 histories")
        healthy = x[batch.normal_rows()]
        self.mean = healthy.mean(axis=(0, 1))
        self.scale = np.maximum(healthy.std(axis=(0, 1)), 1e-6)
        labels = health_to_classes(health)
        if not np.array_equal(np.unique(labels), np.arange(5)):
            raise ValueError("All five classes are required")
        if self.method == "et":
            self.model = ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2,
                                             n_jobs=self.threads, random_state=self.seed)
        elif self.method == "svm":
            self.model = SVC(kernel="rbf", C=1., gamma="scale", cache_size=1024,
                             decision_function_shape="ovr", probability=False,
                             random_state=self.seed)
        else:
            self.model = HistGradientBoostingClassifier(**HGB_PARAMETERS,
                                                        random_state=self.seed)
        weights = binary_flight_weights(health, groups) * len(health)
        with threadpool_limits(limits=self.threads):
            self.model.fit(self._values(x), labels, sample_weight=weights)
        return self

    def _values(self, windows):
        x = validate_windows(windows)
        if x.shape[1:] != (16, 10):
            raise ValueError("Expected the paper's 16 by 10 histories")
        return ((x - self.mean) / self.scale).astype(np.float32).reshape(len(x), 160)

    def score(self, windows):
        x = self._values(windows)
        if not len(x):
            return ScoreBatch(np.empty(0), np.empty((0, 4)))
        if self.method == "svm":
            margins = self.model.decision_function(x)
            motors = margins[:, 1:] - margins[:, :1]
            return ScoreBatch(motors.max(axis=1), motors)
        p = self.model.predict_proba(x)
        return ScoreBatch(p[:, 1:].sum(axis=1), p[:, 1:])
