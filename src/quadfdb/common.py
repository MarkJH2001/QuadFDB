"""Paper input contracts and unchanged numerical/weighting helpers.

Extracted from the internal numerical benchmark; see protocol/source_manifest.json.
No raw data, labels from held-out flights, or platform services are loaded here.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import random
import numpy as np
CLASS_NAMES = ("healthy", "M1", "M2", "M3", "M4")
FEATURE_NAMES = (
    "position_relative_x_m", "position_relative_y_m", "position_relative_z_m",
    "gyro_p_rad_s", "gyro_q_rad_s", "gyro_r_rad_s",
    "command_pre_pwm_m1_fraction", "command_pre_pwm_m2_fraction",
    "command_pre_pwm_m3_fraction", "command_pre_pwm_m4_fraction",
)

PHYSICAL_FEATURE_NAMES = FEATURE_NAMES + (
    "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w",
    "acceleration_body_x_m_s2", "acceleration_body_y_m_s2",
    "acceleration_body_z_m_s2", "battery_voltage_V",
)

def validate_windows(windows: np.ndarray, *, feature_names=None) -> np.ndarray:
    values = np.asarray(windows)
    names = FEATURE_NAMES if feature_names is None else tuple(feature_names)
    if names not in (FEATURE_NAMES, PHYSICAL_FEATURE_NAMES):
        raise ValueError("Unknown benchmark input contract")
    if values.ndim != 3 or values.shape[1] < 2 or values.shape[2] != len(names):
        raise ValueError(f"Expected finite [windows, history>=2, {len(names)}] past-only inputs")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite algorithm input")
    return values.astype(np.float32, copy=False)

@dataclass(frozen=True)
class FitBatch:
    """Audited within-flight windows. Labels describe health, not fault strength.

    The caller selects fit flights and complete histories before making this
    object. No telemetry phase or fault metadata becomes a model input.
    """

    windows: np.ndarray
    motor_health: np.ndarray | None = None
    flight_ids: np.ndarray | None = None
    healthy_mask: np.ndarray | None = None
    fault_age_ms: np.ndarray | None = None
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self):
        windows = validate_windows(self.windows, feature_names=self.feature_names)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        if not len(windows):
            raise ValueError("Fit windows must not be empty")
        object.__setattr__(self, "windows", windows)
        n = len(windows)
        if self.motor_health is not None:
            health = np.asarray(self.motor_health, dtype=np.float32)
            if health.shape != (n, 4) or not np.isfinite(health).all():
                raise ValueError("Expected finite motor_health [windows,4]")
            if np.any((health < 0) | (health > 1)):
                raise ValueError("Motor health labels must be in [0,1]")
            object.__setattr__(self, "motor_health", health)
        if self.flight_ids is not None:
            groups = np.asarray(self.flight_ids)
            if groups.shape != (n,):
                raise ValueError("Expected one flight_id per fit window")
            object.__setattr__(self, "flight_ids", groups)
        if self.healthy_mask is not None:
            mask = np.asarray(self.healthy_mask)
            if mask.shape != (n,) or mask.dtype != np.bool_:
                raise ValueError("Expected boolean healthy_mask [windows]")
            if self.motor_health is not None and np.any(mask & (self.motor_health < 0.5).any(axis=1)):
                raise ValueError("A fault label cannot enter healthy-only fitting")
            object.__setattr__(self, "healthy_mask", mask)
        if self.fault_age_ms is not None:
            if self.motor_health is None:
                raise ValueError("Fault age supervision requires motor_health labels")
            ages = np.asarray(self.fault_age_ms, dtype=np.float64)
            healthy = (self.motor_health >= 0.5).all(axis=1)
            if (ages.shape != (n,) or not np.isnan(ages[healthy]).all()
                    or not np.isfinite(ages[~healthy]).all() or np.any(ages[~healthy] < 0)):
                raise ValueError("Fault age must be NaN for healthy labels and finite nonnegative for faults")
            # Supervision from accepted MCU timestamps is available only at
            # fitting. It is never part of the ten-channel score input.
            object.__setattr__(self, "fault_age_ms", ages)

    def normal_rows(self) -> np.ndarray:
        if self.healthy_mask is not None:
            mask = self.healthy_mask
        elif self.motor_health is not None:
            mask = (self.motor_health >= 0.5).all(axis=1)
        else:
            raise ValueError("Healthy-only fitting requires healthy_mask or motor_health")
        if not mask.any():
            raise ValueError("Healthy fit windows are required")
        return mask

    def require_supervision(self) -> tuple[np.ndarray, np.ndarray]:
        if self.motor_health is None or self.flight_ids is None:
            raise ValueError("Supervised motor fitting requires motor_health and flight_ids")
        return self.motor_health, self.flight_ids

@dataclass(frozen=True)
class ScoreBatch:
    """Endpoint scores. Larger means more faulty. Unsupported outputs are None."""

    detection_score: np.ndarray
    motor_scores: np.ndarray | None = None
    score_valid: np.ndarray | None = None
    failure_reason: np.ndarray | None = None

    def __post_init__(self):
        score = np.asarray(self.detection_score)
        if score.ndim != 1:
            raise ValueError("Expected detection_score [windows]")
        valid = np.ones(len(score), dtype=bool) if self.score_valid is None else np.asarray(self.score_valid)
        if valid.shape != score.shape or valid.dtype != np.bool_:
            raise ValueError("Expected boolean score_valid [windows]")
        if not np.isfinite(score[valid]).all() or not np.isnan(score[~valid]).all():
            raise ValueError("Valid detection scores must be finite; failed scores must be NaN")
        reasons = np.full(len(score), "", dtype="U256") if self.failure_reason is None else np.asarray(self.failure_reason)
        if reasons.shape != score.shape or reasons.dtype.kind not in "US":
            raise ValueError("Expected failure_reason string vector [windows]")
        empty_reason = b"" if reasons.dtype.kind == "S" else ""
        if np.any(reasons[valid] != empty_reason) or np.any(reasons[~valid] == empty_reason):
            raise ValueError("Failed scores require reasons; valid scores must have empty reasons")
        object.__setattr__(self, "detection_score", score)
        object.__setattr__(self, "score_valid", valid)
        object.__setattr__(self, "failure_reason", reasons)
        if self.motor_scores is not None:
            motors = np.asarray(self.motor_scores)
            if (motors.shape != (len(score), 4) or not np.isfinite(motors[valid]).all()
                    or not np.isnan(motors[~valid]).all()):
                raise ValueError("Expected motor_scores [windows,4], finite when valid and NaN on failure")
            object.__setattr__(self, "motor_scores", motors)

    def as_legacy_tuple(self) -> tuple[np.ndarray, np.ndarray | None]:
        if not self.score_valid.all():
            raise ValueError("Legacy tuple cannot represent failed endpoint scores")
        return self.detection_score, self.motor_scores

def ridge_operator(design, target, ridge):
    """Mean squared least squares; the final constant is unpenalized."""
    if not np.isfinite(ridge) or ridge <= 0:
        raise ValueError("koopman_ridge must be positive")
    design, target = np.asarray(design, np.float64), np.asarray(target, np.float64)
    penalty = np.eye(design.shape[1]) * ridge
    penalty[-1, -1] = 0
    gram = design.T @ design / len(design)
    result = np.linalg.solve(gram + penalty, design.T @ target / len(design))
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite Koopman operator")
    return result

def health_to_classes(motor_health: np.ndarray) -> np.ndarray:
    """Map healthy/single-fault health targets to the five physical classes."""
    health = np.asarray(motor_health, dtype=np.float32)
    if health.ndim != 2 or health.shape[1] != 4 or not np.isfinite(health).all():
        raise ValueError("Expected finite motor_health [N,4]")
    if np.any((health < 0) | (health > 1)):
        raise ValueError("Motor health must be in [0,1]")
    faults = health < 0.5
    if np.any(faults.sum(axis=1) > 1):
        raise ValueError("Five-class curve control requires at most one faulty motor")
    labels = np.zeros(len(health), dtype=np.int64)
    positive = faults.any(axis=1)
    labels[positive] = np.argmin(health[positive], axis=1) + 1
    return labels

def _class_flight_weights(classes: np.ndarray, flight_ids: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Healthy/fault 50% each; equal fault-motor and flight mass within class."""
    classes, groups = np.asarray(classes), np.asarray(flight_ids)
    if classes.ndim != 1 or groups.shape != classes.shape or not len(classes):
        raise ValueError("Expected nonempty class and flight vectors of equal length")
    if not np.issubdtype(classes.dtype, np.integer) or np.any((classes < 0) | (classes > 4)):
        raise ValueError("Class indices must be integers in [0,4]")
    present = np.unique(classes)
    fault_classes=present[present!=0]
    class_mass={int(c):(0.5 if c==0 else 0.5/len(fault_classes)) for c in present}
    if 0 not in present or not len(fault_classes):
        class_mass={int(c):1.0/len(present) for c in present}
    weights = np.zeros(len(classes), dtype=np.float64)
    rows = []
    for cls in present:
        flights = np.unique(groups[classes == cls])
        mass = class_mass[int(cls)] / len(flights)
        for flight in flights:
            mask = (classes == cls) & (groups == flight)
            count = int(mask.sum())
            if not count:
                raise ValueError("Flight identifiers must have stable reflexive equality")
            weights[mask] = mass / count
            rows.append({"class_index": int(cls), "class_name": CLASS_NAMES[int(cls)],
                         "flight_id": str(flight), "window_count": count,
                         "class_total_sampling_mass": class_mass[int(cls)],
                         "flight_within_class_sampling_mass": mass,
                         "per_window_sampling_mass": mass / count})
    weights /= weights.sum()
    return weights, rows

def binary_flight_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Balance the event classes, then the flights within each event class."""
    positive = (y < 0.5).any(axis=1)
    weights = np.zeros(len(y), dtype=np.float64)
    for cls in (False, True):
        ids = np.unique(groups[positive == cls])
        for group in ids:
            mask = (positive == cls) & (groups == group)
            weights[mask] = 0.5 / len(ids) / mask.sum()
    return weights

def class_flight_weights(targets, groups):
    if not np.array_equal(np.unique(targets), np.arange(5)):
        raise ValueError("Fitting requires healthy and each motor fault class")
    return _class_flight_weights(targets, groups)[0]


def seed_neural(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
