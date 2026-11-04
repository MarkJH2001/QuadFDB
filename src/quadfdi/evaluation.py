"""Healthy-CAL q99 threshold and the unchanged native event evaluator."""
import numpy as np
from .metrics import evaluate_real_episodes


def calibrate_q99(healthy_scores):
    """Caller supplies only eligible, valid healthy CAL endpoint scores."""
    values = np.asarray(healthy_scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Nonempty finite healthy CAL scores are required")
    return float(np.quantile(values, .99, method="higher"))


def evaluate(episodes, threshold):
    return evaluate_real_episodes(episodes, threshold, min_consecutive=2,
                                  expected_endpoint_period_s=.02)
