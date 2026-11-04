"""Neural architectures used in the QuadFDB paper comparison.

Extracted from the study implementations in
``CrazyFDD/src/crazyfdd/algorithms/{expanded_neural,training}.py`` and
``CrazyFDD/src/crazyfdd/model.py``. The four-output references predict
binary motor health: one is healthy and zero is faulty. Their benchmark
motor fault scores are therefore ``1 - health``.

The LSTM follows the independently implemented Garg et al. detector
architecture, with author-source reference commit
``733e109d0d143cc8e7a20afb21f3f74c5bf794b7``. Hidden and cell states reset
for each independent window. This is a declared adaptation of that work,
not a claim to reproduce its original simulation experiment or codebase.

Torch is imported only when constructing a network. No pretrained model,
PyOD estimator, original research checkout, or flight data is imported.
"""

from __future__ import annotations


def build_network(method: str, window_size: int, config: dict | None = None):
    """Build an MLP, FCN, LSTM, TCN, or Transformer health reference.

    Input shape is ``[batch, window_size, 10]``. All outputs have shape
    ``[batch, 4]`` and use sigmoid activation. Seed the RNG *before* calling
    this function when reproducing initialization and training.
    """
    import torch
    from torch import nn

    if isinstance(window_size, bool) or not isinstance(window_size, int) or window_size < 2:
        raise ValueError("window_size must be an integer of at least two")
    method = method.lower().removesuffix("_bce")
    config = {} if config is None else config
    hidden = int(config.get("hidden_size", 64))

    class LSTMHealth(nn.Module):
        def __init__(self):
            super().__init__()
            self.LSTM = nn.LSTM(10, 128, batch_first=True)
            # The study preserves this unused layer for state-dict and RNG
            # initialization compatibility with its Garg architecture.
            self.fc1 = nn.Linear(128, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 4)
            self.activation = nn.ReLU()
            self.output_activation = nn.Sigmoid()

        def forward(self, x):
            if x.ndim != 3 or tuple(x.shape[1:]) != (window_size, 10):
                raise ValueError("Wrong LSTM window shape")
            sequence, _ = self.LSTM(x)
            value = self.activation(sequence[:, -1, :])
            value = self.activation(self.fc2(value))
            return self.output_activation(self.fc3(value))

    class CausalBlock(nn.Module):
        def __init__(self, inp, out, dilation):
            super().__init__()
            self.left = 2 * dilation
            self.conv1 = nn.Conv1d(inp, out, 3, dilation=dilation)
            self.conv2 = nn.Conv1d(out, out, 3, dilation=dilation)
            self.skip = nn.Conv1d(inp, out, 1) if inp != out else nn.Identity()

        def forward(self, x):
            value = torch.relu(self.conv1(nn.functional.pad(x, (self.left, 0))))
            value = torch.relu(self.conv2(nn.functional.pad(value, (self.left, 0))))
            return torch.relu(value + self.skip(x))

    class TCNHealth(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.Sequential(CausalBlock(10, 32, 1), CausalBlock(32, 32, 2))
            self.head = nn.Linear(32, 4)

        def forward(self, x):
            if tuple(x.shape[1:]) != (window_size, 10):
                raise ValueError("Wrong TCN window shape")
            return torch.sigmoid(self.head(self.blocks(x.transpose(1, 2))[:, :, -1]))

    class FCNHealth(nn.Module):
        def __init__(self):
            super().__init__()
            layers, inp = [], 10
            for out, kernel in ((128, 8), (256, 5), (128, 3)):
                layers.extend((nn.Conv1d(inp, out, kernel, padding="same"),
                               nn.BatchNorm1d(out), nn.ReLU()))
                inp = out
            self.blocks = nn.Sequential(*layers)
            self.head = nn.Linear(128, 4)

        def forward(self, x):
            return torch.sigmoid(self.head(self.blocks(x.transpose(1, 2)).mean(dim=-1)))

    class TransformerHealth(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Linear(10, hidden)
            self.position = nn.Parameter(torch.empty(1, window_size, hidden))
            nn.init.normal_(self.position, std=.02)
            layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 4, dropout=.1,
                                                batch_first=True, activation="gelu")
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.head = nn.Linear(hidden, 4)

        def forward(self, x):
            # The whole input window lies at or before the endpoint. Attention
            # within that window does not access a subsequent scored window.
            value = self.encoder(self.input(x) + self.position)[:, -1]
            return torch.sigmoid(self.head(value))

    if method == "mlp":
        return nn.Sequential(nn.Flatten(), nn.Linear(window_size * 10, 128), nn.ReLU(),
                             nn.Dropout(.2), nn.Linear(128, 64), nn.ReLU(),
                             nn.Linear(64, 4), nn.Sigmoid())
    if method == "fcn":
        return FCNHealth()
    if method == "lstm":
        return LSTMHealth()
    if method == "tcn":
        return TCNHealth()
    if method == "transformer":
        return TransformerHealth()
    raise ValueError(f"Unknown neural reference: {method}")


def build_control_network():
    """Build the distinct 201→128→64→5 categorical ablation head.

    Outputs are logits. Use categorical cross-entropy during fitting and
    softmax for inference. This is not the main four-output BCE MLP.
    """
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(201, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 64), torch.nn.ReLU(), torch.nn.Linear(64, 5),
    )
