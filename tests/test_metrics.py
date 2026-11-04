"""Event timing, censoring, alarm continuity and score ranking regressions."""

import copy
import json
import unittest

import numpy as np

from quadfdi.metrics import evaluate_real_episodes


def fixture(case="step", *, samples=41, onset=0.2, end=0.5, motor=2):
    time = np.arange(samples) * 0.02
    event = (time >= onset - 1e-9) & (time < end - 1e-9) if case not in ("healthy", "sham") else np.zeros(samples, bool)
    healthy = time < onset - 1e-9 if case not in ("healthy", "sham") else np.ones(samples, bool)
    return {"time_s": time, "window_end_steps": np.arange(samples),
            "event_mask": event, "healthy_exposure_mask": healthy,
            "observation_valid": np.ones(samples, bool), "scores": np.zeros(samples),
            "metadata": {"case": case, "motor": motor, "alpha": 0.9,
                         "onset_s": onset, "end_s": end, "role": "test", "trial_id": case}}


class RealMetricsTests(unittest.TestCase):
    def evaluate(self, episodes, **kwargs):
        return evaluate_real_episodes(episodes, 0.5, **kwargs)

    def test_two_endpoint_confirmation_has_causal_delay(self):
        episode = fixture()
        episode["scores"][13:15] = 0.9  # 0.26 s and 0.28 s.
        report = self.evaluate([episode])
        row = report["episodes"][0]
        self.assertTrue(row["new_alarm_detected"])
        self.assertAlmostEqual(row["first_new_alarm_delay_s"], 0.08)
        self.assertAlmostEqual(row["first_new_alarm_time_s"], 0.28)
        self.assertFalse(row["missed_detection"])
        self.assertEqual(report["events"]["new_alarm_detections"], 1)
        self.assertEqual(report["alarm_rule"]["min_consecutive"], 2)
        self.assertFalse(report["point_metrics"]["point_adjustment"])
        json.dumps(report, allow_nan=False)

    def test_exact_end_confirmation_is_not_active_event_detection(self):
        episode = fixture()
        episode["scores"][24:26] = 0.9  # Confirm at actual end, 0.50 s.
        row = self.evaluate([episode])["episodes"][0]
        self.assertFalse(row["active_event_alarm_hit"])
        self.assertTrue(row["missed_detection"])
        self.assertFalse(row["right_censored"])

    def test_carried_alarm_is_a_hit_but_not_a_new_detection(self):
        episode = fixture()
        episode["scores"][:] = 0.9
        report = self.evaluate([episode])
        row = report["episodes"][0]
        self.assertTrue(row["preexisting_alarm_at_onset"])
        self.assertTrue(row["active_event_alarm_hit"])
        self.assertAlmostEqual(row["first_active_alarm_delay_s"], 0.0)
        self.assertFalse(row["new_alarm_detected"])
        self.assertTrue(row["missed_detection"])
        self.assertEqual(report["events"]["preexisting_only_alarm_hits"], 1)

    def test_prefault_first_vote_does_not_become_a_fresh_event_alarm(self):
        episode = fixture()
        episode["scores"][9:12] = 0.9
        row = self.evaluate([episode])["episodes"][0]
        self.assertFalse(row["preexisting_alarm_at_onset"])
        self.assertTrue(row["preexisting_score_above_threshold_at_onset"])
        self.assertTrue(row["active_event_alarm_hit"])
        self.assertFalse(row["new_alarm_detected"])

    def test_skipped_endpoint_resets_confirmation(self):
        episode = fixture()
        ends = np.array([0, 1, 9, 10, 12, 13])
        episode["window_end_steps"] = ends
        episode["scores"] = np.array([0, 0, 0, 0.9, 0.9, 0.9])
        row = self.evaluate([episode])["episodes"][0]
        self.assertAlmostEqual(row["first_new_alarm_time_s"], 0.26)
        self.assertFalse(row["event_scored_complete"])

    def test_invalid_endpoint_resets_confirmation(self):
        episode = fixture()
        episode["scores"][10:14] = 0.9
        episode["observation_valid"][11] = False
        row = self.evaluate([episode])["episodes"][0]
        self.assertAlmostEqual(row["first_new_alarm_time_s"], 0.26)
        self.assertFalse(row["event_scored_complete"])

    def test_incomplete_history_is_invalid_even_with_valid_last_observation(self):
        episode = fixture()
        episode["scores"][10:15] = 0.9
        episode["window_input_valid"] = np.ones(41, bool)
        episode["window_input_valid"][11:13] = False
        row = self.evaluate([episode])["episodes"][0]
        self.assertAlmostEqual(row["first_new_alarm_time_s"], 0.28)
        self.assertFalse(row["event_scored_complete"])

    def test_held_prefault_window_without_event_supervision_cannot_confirm(self):
        episode = fixture()
        episode["scores"][10:15] = 0.9
        episode["window_supervision_valid"] = np.ones(41, bool)
        episode["window_supervision_valid"][10:12] = False
        row = self.evaluate([episode])["episodes"][0]
        self.assertAlmostEqual(row["first_new_alarm_time_s"], 0.24)
        self.assertTrue(row["event_scored_complete"])
        self.assertEqual(row["label_only_event_gap_endpoint_count"], 2)

    def test_unlabeled_onset_does_not_restart_an_existing_alarm_run(self):
        episode = fixture()
        episode["event_mask"][10] = False
        episode["healthy_exposure_mask"][10] = False
        episode["window_supervision_valid"] = np.ones(41, bool)
        episode["window_supervision_valid"][10:12] = False
        episode["scores"][:] = 1
        report = self.evaluate([episode])
        row = report["episodes"][0]
        self.assertTrue(row["preexisting_alarm_at_onset"])
        self.assertTrue(row["active_event_alarm_hit"])
        self.assertFalse(row["new_alarm_detected"])
        self.assertTrue(row["event_scored_complete"])
        self.assertFalse(row["right_censored"])
        self.assertEqual(report["events"]["new_alarm_detections"], 0)
        self.assertEqual(report["events"]["preexisting_alarm_episodes"], 1)

    def test_initial_target_label_delay_does_not_censor_constant_zero_detector(self):
        episode = fixture()
        episode["event_mask"][10] = False
        episode["healthy_exposure_mask"][10] = False
        episode["window_supervision_valid"] = np.ones(41, bool)
        episode["window_supervision_valid"][10:12] = False
        report = self.evaluate([episode])
        row = report["episodes"][0]
        self.assertTrue(row["event_scored_complete"])
        self.assertTrue(row["missed_detection"])
        self.assertFalse(row["right_censored"])
        self.assertEqual(report["event_horizons"]["100_ms"]["censored_episodes"], 0)
        self.assertEqual(report["event_horizons"]["200_ms"]["censored_episodes"], 0)

    def test_nonhover_context_inside_event_resets_confirmation(self):
        episode = fixture()
        episode["scores"][10:14] = 0.9
        episode["audit_phase"] = np.full(41, "fault_window")
        episode["audit_phase"][11] = "landing"
        row = self.evaluate([episode])["episodes"][0]
        self.assertAlmostEqual(row["first_new_alarm_time_s"], 0.26)
        self.assertFalse(row["event_scored_complete"])

    def test_exact_onset_exclusion_is_not_missing_postonset_coverage(self):
        episode = fixture()
        episode["event_mask"][10] = False
        episode["healthy_exposure_mask"][10] = False
        episode["window_supervision_valid"] = np.ones(41, bool)
        episode["window_supervision_valid"][10] = False
        report = self.evaluate([episode])
        row = report["episodes"][0]
        self.assertTrue(row["event_observation_complete"])
        self.assertTrue(row["event_scored_complete"])
        self.assertTrue(row["missed_detection"])
        self.assertEqual(report["events"]["misses_fully_observed"], 1)
        self.assertEqual(report["event_horizons"]["100_ms"]["observed_episodes"], 1)

    def test_nonhover_alarm_does_not_carry_into_measured_healthy_interval(self):
        episode = fixture(case="healthy")
        episode["healthy_exposure_mask"][:10] = False
        episode["window_healthy_exposure_mask"] = episode["healthy_exposure_mask"].copy()
        episode["scores"][:] = 0.9
        report = self.evaluate([episode])
        self.assertEqual(report["healthy"]["false_alarm_transitions"], 1)
        self.assertAlmostEqual(report["healthy"]["evaluated_healthy_exposure_s"], 0.6)
        self.assertEqual(report["episodes"][0]["healthy_alarm_endpoint_count"], 30)

    def test_postrelease_scores_never_become_fault_or_healthy_alarms(self):
        episode = fixture()
        episode["scores"][25:] = 0.9
        report = self.evaluate([episode])
        self.assertFalse(report["episodes"][0]["active_event_alarm_hit"])
        self.assertEqual(report["healthy"]["false_alarm_transitions"], 0)

    def test_incorrect_time_spacing_cannot_confirm_alarm(self):
        episode = fixture(samples=5, onset=0, end=0.1)
        episode["time_s"] = np.array([0, 0.02, 0.045, 0.065, 0.085])
        episode["scores"][1:3] = 0.9
        row = self.evaluate([episode])["episodes"][0]
        self.assertFalse(row["active_event_alarm_hit"])

    def test_right_censored_nondetection_is_not_a_miss(self):
        episode = fixture(samples=17)
        report = self.evaluate([episode])
        row = report["episodes"][0]
        self.assertTrue(row["right_censored"])
        self.assertIsNone(row["missed_detection"])
        self.assertEqual(report["events"]["misses_fully_observed"], 0)
        self.assertEqual(report["event_horizons"]["100_ms"]["observed_episodes"], 1)
        self.assertEqual(report["event_horizons"]["200_ms"]["censored_episodes"], 1)

    def test_horizons_report_actual_detection_and_coverage(self):
        episode = fixture()
        episode["scores"][16:18] = 0.9  # Confirmation at age 140 ms.
        report = self.evaluate([episode])
        self.assertEqual(report["event_horizons"]["100_ms"]["new_alarm_detection_percentage"], 0)
        self.assertEqual(report["event_horizons"]["200_ms"]["new_alarm_detection_percentage"], 100)

    def test_horizon_exact_deadline_requires_valid_input_decision(self):
        episode = fixture()
        episode["window_input_valid"] = np.ones(41, bool)
        episode["window_input_valid"][15] = False  # Exact 100 ms deadline.
        report = self.evaluate([episode])
        self.assertEqual(report["event_horizons"]["100_ms"]["censored_episodes"], 1)
        self.assertEqual(report["event_horizons"]["100_ms"]["observed_episodes"], 0)

    def test_event_end_exclusivity_wins_when_horizon_equals_actual_end(self):
        episode = fixture(end=0.4)
        episode["window_input_valid"] = np.ones(41, bool)
        episode["window_input_valid"][20] = False  # End equals 200 ms deadline.
        report = self.evaluate([episode])
        self.assertTrue(report["episodes"][0]["event_scored_complete"])
        self.assertEqual(report["event_horizons"]["200_ms"]["observed_episodes"], 1)

    def test_short_completed_event_is_separate_from_horizon_censoring(self):
        episode = fixture(case="finite_shutdown", end=0.22)
        report = self.evaluate([episode])
        self.assertFalse(report["episodes"][0]["right_censored"])
        self.assertEqual(report["event_horizons"]["100_ms"]["events_ended_before_horizon"], 1)
        self.assertEqual(report["event_horizons"]["100_ms"]["censored_episodes"], 0)
        self.assertIsNone(report["event_horizons"]["100_ms"]["new_alarm_detection_percentage"])

    def test_healthy_exposure_uses_observed_intervals_and_never_bridges_gap(self):
        episode = fixture(case="healthy", samples=5)
        episode["observation_valid"][2] = False
        episode["scores"][:] = 0.9
        report = self.evaluate([episode])
        self.assertAlmostEqual(report["healthy"]["observed_healthy_exposure_s"], 0.04)
        self.assertAlmostEqual(report["healthy"]["evaluated_healthy_exposure_s"], 0.04)
        self.assertEqual(report["healthy"]["false_alarm_transitions"], 2)
        self.assertEqual(report["healthy"]["healthy_episode_false_alarm_rate"], 1)
        self.assertEqual(report["events"]["fault_episodes"], 0)

    def test_threshold_equality_does_not_alarm_and_sham_has_no_fault_event(self):
        episode = fixture(case="sham")
        episode["scores"][:] = 0.5
        report = self.evaluate([episode])
        self.assertEqual(report["healthy"]["false_alarm_transitions"], 0)
        self.assertEqual(report["events"]["fault_episodes"], 0)

    def test_point_metrics_use_raw_scores_with_tie_groups_and_flight_weights(self):
        positive = fixture(samples=2, onset=0, end=0.04)
        positive["scores"][:] = 0.5
        negative = fixture(case="healthy", samples=8)
        negative["scores"][:] = 0.5
        point = self.evaluate([positive, negative])["point_metrics"]
        self.assertAlmostEqual(point["flight_balanced"]["auroc"], 0.5)
        self.assertAlmostEqual(point["pooled_frames"]["auroc"], 0.5)
        self.assertAlmostEqual(point["flight_balanced"]["average_precision"], 0.5)
        self.assertAlmostEqual(point["pooled_frames"]["average_precision"], 0.2)

    def test_point_ranking_matches_known_auroc_and_average_precision(self):
        episode = fixture(samples=4, onset=0, end=0.08)
        episode["event_mask"] = np.array([False, True, False, True])
        episode["healthy_exposure_mask"] = ~episode["event_mask"]
        episode["scores"] = np.array([0.1, 0.4, 0.35, 0.8])
        report = self.evaluate([episode])["point_metrics"]["pooled_frames"]
        self.assertEqual(report["auroc"], 1.0)
        self.assertEqual(report["average_precision"], 1.0)
        episode["scores"] = np.array([0.1, 0.35, 0.4, 0.8])
        report = self.evaluate([episode])["point_metrics"]["pooled_frames"]
        self.assertEqual(report["auroc"], 0.75)
        self.assertAlmostEqual(report["average_precision"], 5 / 6)

    def test_empty_window_output_is_unscored_and_not_a_detection_miss(self):
        episode = fixture()
        episode["window_end_steps"] = []
        episode["scores"] = []
        report = self.evaluate([episode])
        self.assertIsNone(report["episodes"][0]["missed_detection"])
        self.assertEqual(report["events"]["no_valid_active_event_endpoints"], 1)
        self.assertEqual(report["point_metrics"]["pooled_frames"]["status"], "unavailable")

    def test_supervision_and_unlabeled_response_are_excluded_from_point_metrics(self):
        episode = fixture()
        episode["window_supervision_valid"] = np.ones(len(episode["time_s"]), bool)
        episode["window_supervision_valid"][11] = False
        report = self.evaluate([episode])
        self.assertEqual(report["point_metrics"]["pooled_frames"]["positive_endpoints"], 14)
        self.assertEqual(report["point_metrics"]["pooled_frames"]["negative_endpoints"], 10)

    def test_motor_isolation_is_unavailable_for_anomaly_only_detector(self):
        episode = fixture()
        episode["scores"][11:13] = 0.9
        report = self.evaluate([episode])
        self.assertEqual(report["episodes"][0]["first_motor_diagnosis"], "unavailable")
        self.assertEqual(report["events"]["motor_diagnosis"]["status"], "unavailable")
        self.assertIsNone(report["events"]["motor_diagnosis"]["first_alarm_diagnosis_counts"])
        self.assertIsNone(report["events"]["motor_diagnosis"]["accuracy_given_unambiguous_first_diagnosis"])

    def test_first_motor_diagnosis_gets_one_vote_and_later_correction_does_not_replace_it(self):
        episode = fixture()
        episode["scores"][11:20] = 0.9
        episode["motor_scores"] = np.zeros((41, 4))
        episode["motor_scores"][12, 0] = 1
        episode["motor_scores"][13:, 1] = 1
        report = self.evaluate([episode])
        self.assertEqual(report["episodes"][0]["first_motor"], 1)
        self.assertEqual(report["episodes"][0]["first_motor_diagnosis"], "wrong")
        self.assertEqual(report["events"]["motor_diagnosis"]["first_alarm_diagnosis_counts"]["wrong"], 1)
        self.assertEqual(report["events"]["motor_diagnosis"]["first_alarm_diagnosis_counts"]["correct"], 0)

    def test_tied_motor_scores_are_ambiguous_not_a_motor_one_vote(self):
        episode = fixture(motor=1)
        episode["scores"][11:13] = 0.9
        episode["motor_scores"] = np.zeros((41, 4))
        row = self.evaluate([episode])["episodes"][0]
        self.assertIsNone(row["first_motor"])
        self.assertEqual(row["first_motor_diagnosis"], "ambiguous")

    def test_one_flight_votes_once_and_condition_summary_has_percentages(self):
        first = fixture()
        first["scores"][11:25] = 0.9
        second = fixture()
        second["metadata"]["trial_id"] = "step_second"
        report = self.evaluate([first, second])
        self.assertEqual(report["events"]["new_alarm_detections"], 1)
        self.assertEqual(report["events"]["new_alarm_detection_rate_fully_observed"], 0.5)
        self.assertEqual(report["by_alpha_and_shape"][0]["new_alarm_detection_percentage"], 50)

    def test_invalid_inputs_raise_instead_of_silent_label_or_score_changes(self):
        original = fixture()
        for field, value in (("scores", np.full(41, np.nan)),
                             ("window_end_steps", np.array([0, 0])),
                             ("event_mask", np.ones(41, bool))):
            with self.subTest(field=field):
                episode = copy.deepcopy(original)
                episode[field] = value
                with self.assertRaises(ValueError):
                    self.evaluate([episode])
        with self.assertRaises(ValueError):
            self.evaluate([original], min_consecutive=0)


if __name__ == "__main__":
    unittest.main()
