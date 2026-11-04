"""Synthetic numerical checks only; no recorded flight or model weights."""
import tempfile
import unittest
import numpy as np

from quadfdb import FitBatch, EvidenceBank, RAGB, RFReference, mask_inputs, calibrate_q99
from quadfdb.common import class_flight_weights, health_to_classes
from quadfdb.local_rf import local_cue_features, physical_summary_features
from quadfdb.response import ResponseModel


def synthetic_fit():
    rng = np.random.default_rng(981)
    values, groups, health = [], [], []
    for flight in range(30):
        motor = flight % 5
        for sample in range(8):
            x = rng.normal(0, .02, (16, 10))
            x[:, 6:] += .4
            target = np.ones(4, np.float32)
            if motor and sample >= 4:
                target[motor - 1] = 0
                x[:, motor - 1] += .05
                x[:, 6 + motor - 1] += .08
            values.append(x)
            groups.append("synthetic-flight-" + str(flight))
            health.append(target)
    return FitBatch(np.asarray(values, np.float32), np.asarray(health), np.asarray(groups))


class RepresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch = synthetic_fit()
        cls.bank = EvidenceBank(seed=20261009, threads=1)
        cls.features = cls.bank.fit_transform(cls.batch)

    def test_predictor_excludes_current_target_and_current_command(self):
        x = self.batch.windows[self.batch.normal_rows(), -11:]
        predictor = ResponseModel().fit(x, 1e-4)
        original = predictor.predict(x[:4])
        changed = x[:4].copy()
        changed[:, -1] += 100
        np.testing.assert_array_equal(original, predictor.predict(changed))

    def test_summary_uses_physical_units_and_only_last_eleven_points(self):
        x = np.zeros((2, 16, 10), np.float32)
        x[:, :, 0] = 2 + 3 * (.02 * np.arange(16))
        summary = physical_summary_features(x).reshape(2, 5, 10)
        np.testing.assert_allclose(summary[:, 4, 0], 3, atol=2e-6)
        changed = x.copy()
        changed[:, :5] += 100
        np.testing.assert_array_equal(local_cue_features(x), local_cue_features(changed))
        self.assertEqual(local_cue_features(x).shape, (2, 160))

    def test_folds_are_whole_flight_and_disjoint(self):
        for key, fit_key, held_key in (("response", "predictor_fit_flights", "evidence_flights"),
                                        ("rf", "classifier_fit_flights", "cue_flights")):
            audit = getattr(self.bank, key).audit
            for row in audit["folds"]:
                self.assertFalse(set(row[fit_key]) & set(row[held_key]))
        self.assertEqual(self.features.shape, (240, 201))
        self.assertFalse(self.bank.audit["CAL_or_TEST_used"])
        self.assertEqual(self.bank.response.transform(self.batch.windows[:2])["residuals"].shape, (2, 6, 6))

    def test_masks_keep_width_and_raw_values(self):
        for mask, r, p in (("raw", False, False), ("residual", True, False),
                           ("posterior", False, True), ("all", True, True)):
            values = mask_inputs(self.features, mask)
            self.assertEqual(values.shape, self.features.shape)
            np.testing.assert_array_equal(values[:, :160], self.features[:, :160])
            if not r:
                self.assertFalse(values[:, 160:196].any())
            if not p:
                self.assertFalse(values[:, 196:].any())

    def test_classflight_masses_and_direct_rf_reference(self):
        labels = health_to_classes(self.batch.motor_health)
        weights = class_flight_weights(labels, self.batch.flight_ids)
        np.testing.assert_allclose([weights[labels == c].sum() for c in range(5)],
                                   [.5, .125, .125, .125, .125])
        direct = RFReference.from_bank(self.bank).score(self.batch.windows[:4])
        p = self.bank.rf.transform(self.batch.windows[:4])
        np.testing.assert_array_equal(direct.motor_scores, p[:, 1:])
        np.testing.assert_array_equal(direct.detection_score, p[:, 1:].sum(1))

    def test_ragb_synthetic_fit_save_reload(self):
        # One tiny synthetic full-pipeline fit; no research dataset is read.
        model = RAGB(seed=20261009, threads=1).fit(self.batch)
        expected = model.score(self.batch.windows[:7])
        self.assertEqual(model.model.n_features_in_, 201)
        self.assertEqual(model.model.n_iter_, 200)
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory)
            restored = RAGB.load(directory).score(self.batch.windows[:7])
        np.testing.assert_array_equal(expected.detection_score, restored.detection_score)
        np.testing.assert_array_equal(expected.motor_scores, restored.motor_scores)

    def test_q99_uses_higher_and_rejects_nan(self):
        self.assertEqual(calibrate_q99([0., .2, .4, .6]), .6)
        with self.assertRaises(ValueError):
            calibrate_q99([0., np.nan])


if __name__ == "__main__":
    unittest.main()
