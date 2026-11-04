"""Fitting helpers extracted from the QuadFDI study implementations.

The BCE references retain the Adam, Torch replacement-sampler, and
DataLoader semantics in ``CrazyFDD/.../algorithms/training.py``. The
categorical control retains the separate AdamW, NumPy replacement-sampler,
full-batch schedule, and gradient clipping in
``CrazyFDD/.../algorithms/frozen_evidence_head_ablation.py``.

These helpers accept already selected fitting windows. They neither read
the dataset nor select partitions, normalize signals, calibrate thresholds,
or access calibration/test records.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def fit_baseline_network(model, x, health, groups, config, device, output):
    """Fit a four-output BCE health reference; return ``(model, history)``.

    The caller must seed before network construction and provide the
    healthy-FIT-standardized windows. Seeding again here would change the
    original initialization/dropout RNG sequence, so only the explicit
    DataLoader/sampler generator is initialized in this helper.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    from .common import binary_flight_weights

    x = np.asarray(x)
    health = np.asarray(health)
    groups = np.asarray(groups)
    if (x.ndim != 3 or x.shape[2] != 10 or not len(x)
            or health.shape != (len(x), 4) or groups.shape != (len(x),)
            or not np.isfinite(x).all() or not np.isfinite(health).all()
            or np.any((health < 0) | (health > 1))):
        raise ValueError("Expected finite windows, four health targets, and one flight per window")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = TensorDataset(torch.from_numpy(x.astype(np.float32)),
                            torch.from_numpy(health.astype(np.float32)))
    generator = torch.Generator().manual_seed(config["seed"])
    sampler = WeightedRandomSampler(torch.from_numpy(binary_flight_weights(health, groups)),
                                    len(health), replacement=True, generator=generator)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        sampler=sampler, generator=generator, num_workers=0)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    criterion = torch.nn.BCELoss()
    history = []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total, count = 0.0, 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            loss = criterion(model(batch_x), batch_y)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch_x)
            count += len(batch_x)
        history.append({"epoch": epoch, "fit_loss": total / count})
    (output / "history.json").write_text(
        json.dumps(history, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return model, history


def predict_network(model, windows, device, batch_size=1024):
    """Return the four health outputs, in independent inference batches."""
    import torch

    windows = np.asarray(windows)
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            values = torch.from_numpy(windows[start:start + batch_size].astype(np.float32)).to(device)
            chunks.append(model(values).cpu().numpy())
    return np.concatenate(chunks) if chunks else np.empty((0, 4), dtype=np.float32)


def fit_control_network(inputs, targets, weights, config, device):
    """Fit the five-class CE ablation head; return ``(model, history)``.

    ``inputs`` must contain the precomputed 201-column bank with excluded
    components already zeroed. All masks retain the same architecture.
    Features/tools are not fitted here. The original schedule draws
    ``ceil(N / batch_size)`` complete replacement batches in each epoch.
    """
    import torch
    from .common import seed_neural
    from .neural import build_control_network

    inputs = np.asarray(inputs)
    targets = np.asarray(targets)
    weights = np.asarray(weights, dtype=np.float64)
    if (inputs.ndim != 2 or inputs.shape[1] != 201 or not len(inputs)
            or targets.shape != (len(inputs),) or targets.dtype.kind not in "iu"
            or weights.shape != (len(inputs),) or not np.isfinite(inputs).all()
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isclose(weights.sum(), 1.) or np.any((targets < 0) | (targets > 4))):
        raise ValueError("Expected a finite 201-column bank, class labels 0..4, and normalized weights")
    seed_neural(config["seed"])
    torch.set_num_threads(int(config["torch_threads"]))
    x = torch.as_tensor(inputs, dtype=torch.float32, device=device)
    y = torch.as_tensor(targets, dtype=torch.long, device=device)
    model = build_control_network().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    generator = np.random.default_rng(config["seed"])
    steps = max(1, int(np.ceil(len(x) / config["batch_size"])))
    history = []
    for epoch in range(config["epochs"]):
        total = 0.
        model.train()
        for _ in range(steps):
            indices = generator.choice(len(x), config["batch_size"], replace=True, p=weights)
            selected = torch.as_tensor(indices, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x[selected]), y[selected])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite control head loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach())
        history.append({"epoch": epoch + 1, "loss": total / steps})
    model.eval()
    return model, history
