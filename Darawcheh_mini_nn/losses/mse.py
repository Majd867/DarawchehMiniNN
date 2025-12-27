from __future__ import annotations
import numpy as np
from ..core.network import Loss

class MeanSquaredError(Loss):
    def __init__(self):
        self.y_pred=None
        self.y_true=None


    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred =y_pred
        self.y_true =y_true
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self) -> np.ndarray:
        assert self.y_pred is not None and self.y_true is not None
        N = self.y_pred.shape[0]
        return (2.0 / N) * (self.y_pred - self.y_true)
