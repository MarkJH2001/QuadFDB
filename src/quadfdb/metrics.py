"""Causal, episode-based metrics for real-flight anomaly and motor scores.

This module uses NumPy only. It does not fit a model or a decision threshold.
The caller must fit the threshold on a separate healthy calibration set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np


_TOLERANCE_S = 1e-6
_CONTROL_CASES = {"healthy", "sham"}


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _delay_summary(values: Sequence[float | None]) -> dict:
    values = np.asarray([value for value in values if value is not None], dtype=float)
    return {"count": int(len(values)),
            "median_ms": float(np.median(values) * 1000) if len(values) else None,
            "p95_ms": float(np.percentile(values, 95) * 1000) if len(values) else None,
            "max_ms": float(np.max(values) * 1000) if len(values) else None}


def _mask(value, length: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,) or not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must be a boolean vector with {length} entries")
    return array.astype(bool)


def _optional_time(value, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite seconds or None")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite seconds or None")
    return value


def _rank_metrics(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> dict:
    """Weighted AUROC and non-interpolated AP, with whole equal-score groups."""
    positives = int(np.count_nonzero(labels))
    negatives = int(len(labels) - positives)
    if not len(labels):
        return {"status": "unavailable", "reason": "no supervised valid endpoints",
                "auroc": None, "average_precision": None, "positive_endpoints": 0,
                "negative_endpoints": 0}
    order = np.argsort(scores, kind="stable")
    ordered_scores, ordered_labels, ordered_weights = scores[order], labels[order], weights[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ordered_scores) != 0) + 1]
    positive_weight = np.add.reduceat(ordered_weights * ordered_labels, starts)
    negative_weight = np.add.reduceat(ordered_weights * ~ordered_labels, starts)
    total_positive, total_negative = float(positive_weight.sum()), float(negative_weight.sum())
    auc = None
    if total_positive and total_negative:
        preceding_negative = np.cumsum(negative_weight) - negative_weight
        auc = float(np.sum(positive_weight * (preceding_negative + 0.5 * negative_weight))
                    / (total_positive * total_negative))
    ap = None
    if total_positive:
        descending_positive, descending_negative = positive_weight[::-1], negative_weight[::-1]
        cumulative_positive = np.cumsum(descending_positive)
        cumulative_total = np.cumsum(descending_positive + descending_negative)
        ap = float(np.sum(descending_positive / total_positive
                          * cumulative_positive / cumulative_total))
    return {"status": "available" if auc is not None else "single_class",
            "auroc": auc, "average_precision": ap, "positive_endpoints": positives,
            "negative_endpoints": negatives, "positive_weight": total_positive,
            "negative_weight": total_negative}


def _confirmed_alarms(above: np.ndarray, valid: np.ndarray, ends: np.ndarray,
                      times: np.ndarray, consecutive: int, period: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    confirmed = np.zeros(len(ends), dtype=bool)
    rising = np.zeros(len(ends), dtype=bool)
    run_start = np.full(len(ends), -1, dtype=int)
    count, start = 0, -1
    for index in range(len(ends)):
        adjacent = (index > 0 and valid[index - 1]
                    and ends[index] == ends[index - 1] + 1
                    and abs(times[index] - times[index - 1] - period) <= _TOLERANCE_S)
        if not valid[index] or not above[index]:
            count, start = 0, -1
            continue
        if not adjacent or count == 0:
            count, start = 1, index
        else:
            count += 1
        confirmed[index] = count >= consecutive
        run_start[index] = start
        rising[index] = confirmed[index] and (not adjacent or not confirmed[index - 1])
    return confirmed, rising, run_start


def _measured_exposure(times: np.ndarray, mask: np.ndarray, period: float,
                       indices: np.ndarray | None = None) -> float:
    if len(times) < 2:
        return 0.0
    differences = np.diff(times)
    adjacent = mask[:-1] & mask[1:] & (np.abs(differences - period) <= _TOLERANCE_S)
    if indices is not None:
        adjacent &= np.diff(indices) == 1
    return float(np.sum(differences[adjacent]))


def _coverage(time: np.ndarray, valid: np.ndarray, decision_valid: np.ndarray,
              begin: float | None, finish: float | None, period: float,
              *, include_finish: bool = False) -> tuple[bool, bool]:
    """Require observed coverage through the end, without filling missing frames."""
    if begin is None or finish is None or finish <= begin:
        return False, False
    first = int(np.searchsorted(time, begin - _TOLERANCE_S, side="left"))
    last = int(np.searchsorted(time, finish - _TOLERANCE_S, side="left"))
    if first >= len(time) or last >= len(time):
        return False, False
    if time[first] - begin > period + _TOLERANCE_S or time[last] - finish > period + _TOLERANCE_S:
        return False, False
    observed = bool(valid[first:last + 1].all()
                    and np.all(np.abs(np.diff(time[first:last + 1]) - period) <= _TOLERANCE_S))
    # An observation exactly at onset has no post-onset response. Its deliberate
    # exclusion from supervision is not missing active-event decision coverage.
    before_finish = time <= finish + _TOLERANCE_S if include_finish else time < finish - _TOLERANCE_S
    active = np.flatnonzero((time > begin + _TOLERANCE_S) & before_finish)
    scored = bool(observed and len(active) and decision_valid[active].all())
    return observed, scored


def _first(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def _motor_at(motor_scores: np.ndarray | None, index: int | None,
              expected_motor: int) -> tuple[int | None, str]:
    if motor_scores is None:
        return None, "unavailable"
    if index is None:
        return None, "missed"
    values = motor_scores[index]
    maximum = np.max(values)
    winners = np.flatnonzero(values == maximum)
    if len(winners) != 1:
        return None, "ambiguous"
    motor = int(winners[0] + 1)
    return motor, "correct" if motor == expected_motor else "wrong"


def _event_summary(rows: Sequence[Mapping]) -> dict:
    complete = [row for row in rows if row["event_scored_complete"]]
    detected = sum(row["new_alarm_detected"] for row in rows)
    hits = sum(row["active_event_alarm_hit"] for row in rows)
    diagnosis_rows = [row for row in rows if row["motor_scores_available"]]
    diagnosis_counts = {state: sum(row["first_motor_diagnosis"] == state for row in diagnosis_rows)
                        for state in ("correct", "wrong", "ambiguous", "missed")}
    diagnosed = diagnosis_counts["correct"] + diagnosis_counts["wrong"]
    diagnosis = {"status": "available" if diagnosis_rows else "unavailable",
                 "fault_episodes_with_motor_scores": len(diagnosis_rows),
                 "fault_episodes_without_motor_scores": len(rows) - len(diagnosis_rows),
                 "first_alarm_diagnosis_counts": diagnosis_counts if diagnosis_rows else None,
                 "correct_first_diagnosis_rate_all_available_fault_episodes":
                     _ratio(diagnosis_counts["correct"], len(diagnosis_rows)),
                 "accuracy_given_unambiguous_first_diagnosis":
                     _ratio(diagnosis_counts["correct"], diagnosed)}
    return {"fault_episodes": len(rows), "active_event_alarm_hits": hits,
            "active_event_alarm_rate": _ratio(hits, len(rows)),
            "new_alarm_detections": detected,
            "new_alarm_detection_rate_all_observed_episodes": _ratio(detected, len(rows)),
            "fully_observed_scored_episodes": len(complete),
            "new_alarm_detection_rate_fully_observed":
                _ratio(sum(row["new_alarm_detected"] for row in complete), len(complete)),
            "misses_fully_observed": sum(row["missed_detection"] is True for row in rows),
            "right_censored_episodes": sum(row["right_censored"] for row in rows),
            "incomplete_decision_coverage_episodes": sum(not row["event_scored_complete"] for row in rows),
            "no_valid_active_event_endpoints": sum(row["active_event_endpoint_count"] == 0 for row in rows),
            "preexisting_alarm_episodes": sum(row["preexisting_alarm_at_onset"] is True for row in rows),
            "unknown_preexisting_alarm_episodes": sum(row["preexisting_alarm_at_onset"] is None for row in rows),
            "preexisting_only_alarm_hits": sum(row["active_event_alarm_hit"] and not row["new_alarm_detected"] for row in rows),
            "first_active_alarm_delay": _delay_summary([row["first_active_alarm_delay_s"] for row in rows]),
            "first_new_alarm_delay": _delay_summary([row["first_new_alarm_delay_s"] for row in rows]),
            "motor_diagnosis": diagnosis}


def evaluate_real_episodes(
    episodes: Sequence[Mapping[str, Any]],
    threshold: float,
    *,
    min_consecutive: int = 2,
    expected_endpoint_period_s: float = 0.02,
    horizons_s: Sequence[float] = (0.1, 0.2),
) -> dict[str, Any]:
    """Evaluate causal windows, with one flight vote for event metrics.

    Each episode supplies ``time_s[T]``, ``window_end_steps[K]``, ``scores[K]``,
    ``event_mask[T]``, ``healthy_exposure_mask[T]`` and ``observation_valid[T]``.
    Optional ``motor_scores[K,4]`` uses larger values for more faulty motors.
    Metadata can be nested under ``metadata`` or provided at the top level.
    It contains ``case``, one-based ``motor``, ``alpha``, ``onset_s``, ``end_s``,
    ``role`` and ``trial_id``. Healthy and sham episodes contain no fault labels.

    Optional ``window_input_valid[K]`` gates complete-history input validity.
    Optional ``window_supervision_valid[K]`` gates point labels and active-event
    hit eligibility. Optional ``window_healthy_exposure_mask[K]`` gates healthy
    decisions and their exposure. Alarm continuation uses complete input
    validity AND (healthy window OR bounded active-event monitoring context).
    It preserves the score run through a label-only onset delay. Optional
    ``monitoring_mask[T]`` or the export's ``audit_phase[T]`` excludes non-hover
    monitoring. Other endpoints reset runs, including descent and post-release.
    Scores are decisions at the causal endpoint, never at the window start.
    Two consecutive valid 20 ms endpoints above threshold confirm an alarm by
    default. Missing or invalid endpoints reset the run. No point adjustment,
    interpolation, latency subtraction or extrapolated exposure is used.

    A carried alarm is an active-event hit, but a fresh event detection requires
    its above-threshold run to start at or after onset. Hit and diagnosis
    endpoints require valid post-onset target observations. Misses are assigned
    only when the whole active event has observed, input-valid score coverage.
    Target-label availability does not determine observation completeness.
    Fixed horizons
    require continuous observed/scored coverage and an event still active at
    that horizon. Shorter completed events are reported separately.
    """
    threshold = float(threshold)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if isinstance(min_consecutive, bool) or not isinstance(min_consecutive, (int, np.integer)) or min_consecutive < 1:
        raise ValueError("min_consecutive must be a positive integer")
    period = float(expected_endpoint_period_s)
    if not math.isfinite(period) or period <= 0:
        raise ValueError("expected_endpoint_period_s must be finite and positive")
    horizons = tuple(float(value) for value in horizons_s)
    if any(not math.isfinite(value) or value <= 0 for value in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError("horizons_s must contain distinct positive finite durations")

    results, point_labels, point_scores, point_weights = [], [], [], []
    for episode_index, episode in enumerate(episodes):
        metadata = dict(episode.get("metadata", episode))
        case = metadata.get("case")
        if not isinstance(case, str) or not case:
            raise ValueError("each episode needs a case")
        time = np.asarray(episode["time_s"], dtype=float)
        if time.ndim != 1 or not len(time) or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
            raise ValueError("time_s must be finite, nonempty and strictly increasing")
        ends = np.asarray(episode["window_end_steps"])
        if ends.ndim != 1 or (len(ends) and not np.issubdtype(ends.dtype, np.integer)) or np.any((ends < 0) | (ends >= len(time))) or np.any(np.diff(ends) <= 0):
            raise ValueError("window_end_steps must be increasing, unique, in-range integer indices")
        ends = ends.astype(np.int64)
        scores = np.asarray(episode["scores"], dtype=float)
        score_valid = _mask(episode.get("score_valid", np.ones(len(ends), bool)), len(ends), "score_valid")
        score_requested = _mask(episode.get("score_requested", np.ones(len(ends), bool)), len(ends), "score_requested")
        if np.any(score_valid & ~score_requested):
            raise ValueError("An unrequested score cannot be valid")
        if (scores.shape != ends.shape or not np.isfinite(scores[score_valid]).all()
                or not np.isnan(scores[~score_valid]).all()):
            raise ValueError("scores must be finite on valid endpoints and NaN on failed or skipped endpoints")
        event = _mask(episode["event_mask"], len(time), "event_mask")
        healthy = _mask(episode["healthy_exposure_mask"], len(time), "healthy_exposure_mask")
        valid = _mask(episode["observation_valid"], len(time), "observation_valid")
        if np.any(event & healthy):
            raise ValueError("event_mask and healthy_exposure_mask must be disjoint")
        is_fault = case not in _CONTROL_CASES
        if not is_fault and event.any():
            raise ValueError("healthy and sham episodes cannot contain positive fault labels")
        onset = _optional_time(metadata.get("onset_s"), "onset_s") if is_fault else None
        end = _optional_time(metadata.get("end_s"), "end_s") if is_fault else None
        if onset is not None and end is not None and end <= onset:
            raise ValueError("end_s must follow onset_s")
        alpha = metadata.get("alpha")
        if alpha is not None:
            alpha = float(alpha)
            if not math.isfinite(alpha) or not 0 <= alpha <= 1:
                raise ValueError("alpha must be a finite command multiplier in [0, 1]")
        motor = metadata.get("motor", 1)
        if isinstance(motor, (bool, np.bool_)) or not isinstance(motor, (int, np.integer)) or not 1 <= motor <= 4:
            raise ValueError("motor must be a one-based integer from 1 to 4")
        motor_scores = None
        if episode.get("motor_scores") is not None:
            motor_scores = np.asarray(episode["motor_scores"], dtype=float)
            if (motor_scores.shape != (len(ends), 4) or not np.isfinite(motor_scores[score_valid]).all()
                    or not np.isnan(motor_scores[~score_valid]).all()):
                raise ValueError("motor_scores must be finite on valid endpoints and NaN otherwise, with shape (K,4)")
        supervision = _mask(episode.get("window_supervision_valid", np.ones(len(ends), bool)), len(ends), "window_supervision_valid")
        model_input_valid = valid[ends] & _mask(episode.get("window_input_valid", np.ones(len(ends), bool)), len(ends), "window_input_valid")
        input_valid = model_input_valid & score_valid
        healthy_window = _mask(episode.get("window_healthy_exposure_mask", healthy[ends]), len(ends), "window_healthy_exposure_mask")
        if np.any(healthy_window & ~healthy[ends]):
            raise ValueError("window healthy exposure must be a subset of endpoint healthy exposure")
        endpoint_time = time[ends]
        if "monitoring_mask" in episode:
            monitoring = _mask(episode["monitoring_mask"], len(time), "monitoring_mask")
        elif "audit_phase" in episode:
            phase = np.asarray(episode["audit_phase"])
            if phase.shape != time.shape:
                raise ValueError("audit_phase must have one phase per grid observation")
            monitoring = np.isin(phase, ("baseline_hover", "fault_window"))
        else:
            monitoring = np.ones(len(time), dtype=bool)
        event_window = event[ends] & supervision
        event_context = np.zeros(len(ends), dtype=bool)
        if onset is None:
            event_window[:] = False
        else:
            event_window &= endpoint_time >= onset - _TOLERANCE_S
            event_context = endpoint_time >= onset - _TOLERANCE_S
        if end is not None:
            event_window &= endpoint_time < end - _TOLERANCE_S
            event_context &= endpoint_time < end - _TOLERANCE_S
        elif onset is not None:
            event_context &= event[ends] | (np.abs(endpoint_time - onset) <= _TOLERANCE_S)
        event_window &= monitoring[ends]
        event_context &= monitoring[ends]
        endpoint_valid = input_valid & monitoring[ends] & (healthy_window | event_context)
        above = scores > threshold
        confirmed, rising, run_start = _confirmed_alarms(above, endpoint_valid, ends, endpoint_time, int(min_consecutive), period)
        supervised = input_valid & supervision & (event[ends] | healthy[ends])
        if supervised.any():
            point_labels.append(event[ends][supervised])
            point_scores.append(scores[supervised])
            point_weights.append(np.full(np.count_nonzero(supervised), 1 / np.count_nonzero(supervised)))
        healthy_scored = input_valid & healthy_window & monitoring[ends]
        false_transitions = rising & healthy_scored
        observed_exposure = _measured_exposure(time, valid & healthy & monitoring, period)
        evaluated_exposure = _measured_exposure(endpoint_time, healthy_scored, period, ends)
        row = {"episode_index": episode_index, "trial_id": metadata.get("trial_id"),
               "role": metadata.get("role"), "case": case, "motor": int(motor),
               "alpha": alpha, "is_fault": is_fault,
               "window_count": len(ends), "input_valid_endpoint_count": int(input_valid.sum()),
               "model_input_valid_endpoint_count": int(model_input_valid.sum()),
               "score_requested_endpoint_count": int(score_requested.sum()),
               "score_valid_endpoint_count": int(score_valid.sum()),
               "score_failure_endpoint_count": int(np.count_nonzero(score_requested & ~score_valid)),
               "score_not_requested_endpoint_count": int(np.count_nonzero(~score_requested)),
               "valid_endpoint_count": int(endpoint_valid.sum()),
               "label_only_event_gap_endpoint_count": int(np.count_nonzero(input_valid & event_context & ~event_window)),
               "supervised_endpoint_count": int(supervised.sum()),
               "observed_healthy_exposure_s": observed_exposure,
               "evaluated_healthy_exposure_s": evaluated_exposure,
               "healthy_false_alarm_transitions": int(false_transitions.sum()),
               "healthy_alarm_endpoint_count": int(np.count_nonzero(confirmed & healthy_scored)),
               "healthy_episode_evaluable": bool(healthy_scored.any()) if not is_fault else False,
               "healthy_episode_false_alarm": bool(np.any(confirmed & healthy_scored)) if not is_fault else None,
               "motor_scores_available": motor_scores is not None}
        if is_fault:
            active = input_valid & event_window
            # Confirmation can occur during a label-only delay. The first
            # eligible event hit keeps that run's original start time; it does
            # not acquire a fabricated new start at the first available label.
            fresh = confirmed & active
            for index in np.flatnonzero(fresh):
                if endpoint_time[run_start[index]] < onset - _TOLERANCE_S:
                    fresh[index] = False
            first_active, first_new = _first(confirmed & active), _first(fresh)
            prior = np.flatnonzero(endpoint_valid & (endpoint_time < onset - _TOLERANCE_S)) if onset is not None else np.array([], dtype=int)
            prior_index = int(prior[-1]) if len(prior) else None
            known_prior = prior_index is not None and onset - endpoint_time[prior_index] <= period + _TOLERANCE_S
            decision_grid = np.zeros(len(time), bool)
            decision_grid[ends] = endpoint_valid
            observed_complete, scored_complete = _coverage(time, valid, decision_grid, onset, end, period)
            last_valid_time = float(time[np.flatnonzero(valid)[-1]]) if valid.any() else None
            right_censored = (onset is None or end is None or last_valid_time is None
                              or last_valid_time < end - _TOLERANCE_S
                              or bool(metadata.get("right_censored", False)))
            first_motor, diagnosis = _motor_at(motor_scores, first_new, int(motor))
            first_active_motor, active_diagnosis = _motor_at(motor_scores, first_active, int(motor))
            row.update(onset_s=onset, end_s=end, event_observation_complete=observed_complete,
                       event_scored_complete=scored_complete and not right_censored,
                       right_censored=bool(right_censored), active_event_endpoint_count=int(active.sum()),
                       active_event_alarm_hit=first_active is not None,
                       new_alarm_detected=first_new is not None,
                       preexisting_alarm_at_onset=bool(confirmed[prior_index]) if known_prior else None,
                       preexisting_score_above_threshold_at_onset=bool(above[prior_index]) if known_prior else None,
                       first_active_alarm_delay_s=float(endpoint_time[first_active] - onset) if first_active is not None else None,
                       first_new_alarm_delay_s=float(endpoint_time[first_new] - onset) if first_new is not None else None,
                       first_alarm_time_s=float(endpoint_time[first_active]) if first_active is not None else None,
                       first_new_alarm_time_s=float(endpoint_time[first_new]) if first_new is not None else None,
                       missed_detection=False if first_new is not None else True if scored_complete and not right_censored else None,
                       first_motor=first_motor, first_motor_diagnosis=diagnosis,
                       first_active_alarm_motor=first_active_motor,
                       first_active_alarm_motor_diagnosis=active_diagnosis)
            row["horizons"] = {}
            for horizon in horizons:
                horizon_name = f"{horizon * 1000:g}_ms"
                deadline = onset + horizon if onset is not None else None
                short = end is not None and deadline is not None and end < deadline - _TOLERANCE_S
                inclusive = end is None or deadline is not None and deadline < end - _TOLERANCE_S
                _, horizon_scored = _coverage(time, valid, decision_grid, onset, deadline, period,
                                              include_finish=inclusive)
                status = "event_ended_before_horizon" if short else "observed" if horizon_scored else "censored"
                eligible = status == "observed"
                row["horizons"][horizon_name] = {
                    "duration_ms": float(horizon * 1000), "status": status,
                    "new_alarm_detected": bool(first_new is not None and endpoint_time[first_new] <= deadline + _TOLERANCE_S) if eligible else None,
                    "active_event_alarm_hit": bool(first_active is not None and endpoint_time[first_active] <= deadline + _TOLERANCE_S) if eligible else None}
        results.append(row)

    labels = np.concatenate(point_labels) if point_labels else np.zeros(0, bool)
    scores = np.concatenate(point_scores) if point_scores else np.zeros(0, float)
    weights = np.concatenate(point_weights) if point_weights else np.zeros(0, float)
    fault_rows = [row for row in results if row["is_fault"]]
    healthy_rows = [row for row in results if not row["is_fault"]]
    evaluable_healthy_rows = [row for row in healthy_rows if row["healthy_episode_evaluable"]]
    exposure = float(sum(row["evaluated_healthy_exposure_s"] for row in results))
    transitions = sum(row["healthy_false_alarm_transitions"] for row in results)
    horizon_summary = {}
    for horizon in horizons:
        name = f"{horizon * 1000:g}_ms"
        items = [row["horizons"][name] for row in fault_rows]
        observed = [item for item in items if item["status"] == "observed"]
        detected = sum(item["new_alarm_detected"] for item in observed)
        horizon_summary[name] = {"duration_ms": float(horizon * 1000),
            "observed_episodes": len(observed), "new_alarm_detections": detected,
            "new_alarm_detection_percentage": 100 * detected / len(observed) if observed else None,
            "censored_episodes": sum(item["status"] == "censored" for item in items),
            "events_ended_before_horizon": sum(item["status"] == "event_ended_before_horizon" for item in items)}
    groups = {}
    for row in fault_rows:
        key = (row["case"], str(row["alpha"]))
        groups.setdefault(key, []).append(row)
    by_condition = [{"case": key[0], "alpha": rows[0]["alpha"],
                     **_event_summary(rows),
                     "new_alarm_detection_percentage": 100 * sum(row["new_alarm_detected"] for row in rows) / len(rows)}
                    for key, rows in sorted(groups.items())]
    return {"schema_version": 1, "metric_kind": "causal_real_flight_episode_evaluation",
            "episode_count": len(results),
            "score_coverage": {
                "requested_endpoints": sum(row["score_requested_endpoint_count"] for row in results),
                "valid_endpoints": sum(row["score_valid_endpoint_count"] for row in results),
                "failure_endpoints": sum(row["score_failure_endpoint_count"] for row in results),
                "not_requested_endpoints": sum(row["score_not_requested_endpoint_count"] for row in results),
                "failure_policy": "retain NaN cache entries; exclude failures from rank and threshold; reset alarm run; retain all fault/control flight denominators"},
            "alarm_rule": {"comparison": "score > threshold", "threshold": threshold,
                "min_consecutive": int(min_consecutive), "endpoint_period_s": period,
                "timing_tolerance_s": _TOLERANCE_S,
                "confirmation_time": "last endpoint in the required run",
                "invalid_or_skipped_endpoint": "reset the run; no bridging",
                "decision_mask": "observation_valid at endpoint AND window_input_valid AND hover monitoring AND (eligible healthy window OR bounded event-time context including exact onset)",
                "label_only_gap": "preserve confirmation state and original above-threshold run start; does not censor input-valid coverage",
                "active_hit_mask": "input-valid monitored event endpoint AND window_supervision_valid; strictly before actual end",
                "outside_decision_mask": "reset; excludes takeoff, descent and post-release",
                "threshold_fit": "caller-supplied; fit only on separate healthy calibration episodes"},
            "point_metrics": {"primary": "flight_balanced",
                "flight_balanced": _rank_metrics(labels, scores, weights),
                "pooled_frames": _rank_metrics(labels, scores, np.ones(len(scores))),
                "weighting": "each flight with supervised endpoints has total weight 1; pooled frames are descriptive",
                "point_adjustment": False},
            "events": _event_summary(fault_rows), "event_horizons": horizon_summary,
            "healthy": {"false_alarm_transitions": transitions,
                "observed_healthy_exposure_s": float(sum(row["observed_healthy_exposure_s"] for row in results)),
                "evaluated_healthy_exposure_s": exposure,
                "false_alarm_events_per_hour": transitions * 3600 / exposure if exposure else None,
                "healthy_control_episodes": len(healthy_rows),
                "evaluable_healthy_control_episodes": len(evaluable_healthy_rows),
                "unevaluable_healthy_control_episodes": len(healthy_rows) - len(evaluable_healthy_rows),
                "healthy_control_episodes_with_alarm": sum(row["healthy_episode_false_alarm"] for row in healthy_rows),
                "healthy_episode_false_alarm_rate": _ratio(sum(row["healthy_episode_false_alarm"] for row in healthy_rows), len(healthy_rows)),
                "exposure_method": "sum adjacent valid healthy 20 ms intervals; no first-sample extrapolation or gap filling"},
            "by_alpha_and_shape": by_condition, "episodes": results,
            "conventions": {"event_vote": "one flight per vote",
                "active_alarm": "confirmed alarm at a valid event endpoint strictly before actual end",
                "new_detection": "first confirmed fresh alarm run starting at or after onset",
                "miss": "no fresh alarm with complete observed and input-valid active-event score coverage; label-only initial delay is not an outage",
                "right_censoring": "unknown actual end or observations ending before actual end; reported separately from misses",
                "motor_diagnosis": "one-based unique argmax at the first fresh alarm; ties are ambiguous; absent output is unavailable",
                "horizons": "continuous observed/input-valid score coverage; include the exact horizon decision if event remains active; actual event end is exclusive",
                "time_basis": "provided decision-grid time; not a radio or compute latency measurement"}}
