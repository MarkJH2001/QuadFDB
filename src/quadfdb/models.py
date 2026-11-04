"""Paper-scoped RAGB, frozen-RF reference and controlled classifier heads."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .common import (FitBatch, ScoreBatch, health_to_classes,
                     class_flight_weights, write_json)
from .features import EvidenceBank, mask_inputs
from .local_rf import LocalMotorCue

HGB_PARAMETERS = dict(loss="log_loss", learning_rate=.1, max_iter=200,
                      max_leaf_nodes=31, min_samples_leaf=20,
                      l2_regularization=0., early_stopping=False)
CONTROL_CONFIG = dict(epochs=50, batch_size=256, learning_rate=.0005,
                      weight_decay=1e-5, torch_threads=4)


class EvidenceClassifier:
    """A classifier fitted to a shared, already cross-fitted 201-column bank.

    Choosing ``head='hgb', mask='all'`` gives RAGB. Other choices implement
    the paper's controlled head/input comparisons. Masks apply at fitting
    and inference; they are not test-time image/input deletion experiments.
    """

    def __init__(self, head="hgb", mask="all", seed=20261009, threads=4):
        if head not in ("hgb", "mlp"):
            raise ValueError("Head must be hgb or categorical mlp")
        mask_inputs(np.empty((0, 201), np.float32), mask)
        self.head, self.mask = head, mask
        self.seed, self.threads = int(seed), int(threads)
        self.fitted = False

    def fit(self, bank: EvidenceBank, fit_features, batch: FitBatch,
            *, device="cpu", output=None):
        from .features import array_sha
        from sklearn.ensemble import HistGradientBoostingClassifier
        from threadpoolctl import threadpool_limits
        if not bank.fitted or bank.seed != self.seed:
            raise ValueError("The bank and classifier must share a pipeline seed")
        for key, array in (("selected_fit_inputs_sha256", batch.windows),
                           ("selected_fit_labels_sha256", batch.motor_health),
                           ("selected_fit_groups_sha256", batch.flight_ids),
                           ("joined_201_inputs_sha256", fit_features)):
            if bank.audit[key] != array_sha(array):
                raise ValueError("Fitting data/features differ from the common bank")
        health, groups = batch.require_supervision()
        labels = health_to_classes(health)
        weights = class_flight_weights(labels, groups)
        values = mask_inputs(fit_features, self.mask)
        self.bank = bank
        if self.head == "hgb":
            self.model = HistGradientBoostingClassifier(
                **HGB_PARAMETERS, random_state=self.seed)
            with threadpool_limits(limits=self.threads):
                self.model.fit(values, labels, sample_weight=weights * len(weights))
            self.history = []
        else:
            from .neural_training import fit_control_network
            self.model, self.history = fit_control_network(
                values, labels, weights,
                {**CONTROL_CONFIG, "seed": self.seed, "torch_threads": self.threads},
                device=device)
        self.device = str(device) if self.head == "mlp" else "cpu"
        self.fitted = True
        if output is not None:
            self.save(output)
        return self

    def score(self, windows):
        if not self.fitted:
            raise ValueError("Fit or load the classifier first")
        values = mask_inputs(self.bank.transform(windows), self.mask)
        if not len(values):
            return ScoreBatch(np.empty(0), np.empty((0, 4)))
        if self.head == "hgb":
            from threadpoolctl import threadpool_limits
            with threadpool_limits(limits=self.threads):
                probability = self.model.predict_proba(values)
        else:
            import torch
            self.model.eval()
            # All windows are already transformed using the common bank.
            parts = []
            with torch.no_grad():
                for start in range(0, len(values), 256):
                    tensor = torch.as_tensor(values[start:start + 256],
                                             dtype=torch.float32, device=self.device)
                    parts.append(torch.softmax(self.model(tensor), -1).cpu().numpy())
            probability = np.concatenate(parts)
        return ScoreBatch(probability[:, 1:].sum(axis=1), probability[:, 1:])

    def save(self, output):
        import joblib
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output / "model.joblib")
        write_json(output / "descriptor.json", {
            "format": "quadfdb-portable-v1", "head": self.head,
            "mask": self.mask, "seed": self.seed,
            "feature_audit": self.bank.audit,
            "note": "Portable snapshot; not the internal server checkpoint format"})

    @classmethod
    def load(cls, output):
        """Load a trusted checkpoint created by this package."""
        import joblib
        obj = joblib.load(Path(output) / "model.joblib")
        if not isinstance(obj, EvidenceClassifier) or not obj.fitted:
            raise ValueError("Unexpected checkpoint")
        return obj


class RAGB(EvidenceClassifier):
    def __init__(self, seed=20261009, threads=4):
        super().__init__(head="hgb", mask="all", seed=seed, threads=threads)

    def fit(self, batch: FitBatch, *, output=None):
        bank = EvidenceBank(self.seed, self.threads)
        features = bank.fit_transform(batch)
        return super().fit(bank, features, batch, output=output)


class RFReference:
    """The same local RF that supplies RAGB's five posterior inputs."""

    def __init__(self, seed=20261009, threads=4):
        self.rf = LocalMotorCue(seed=seed, n_estimators=200, n_jobs=threads)

    def fit(self, batch):
        self.rf.fit(batch)
        return self

    @classmethod
    def from_bank(cls, bank):
        obj = cls(bank.seed, bank.threads)
        obj.rf = bank.rf
        return obj

    def score(self, windows):
        p = self.rf.transform(windows)
        return ScoreBatch(p[:, 1:].sum(axis=1), p[:, 1:])
