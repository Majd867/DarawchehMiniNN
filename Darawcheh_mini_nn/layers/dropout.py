from __future__ import annotations
import numpy as np
from ..core.layer import Layer

class Dropout(Layer):
    def __init__(self, p: float = 0.5, seed: int | None = None):
        assert 0.0 <= p < 1.0
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.p == 0.0:
            self.mask = None
            return x
        keep = 1.0 - self.p
        self.mask = (self.rng.random(size=x.shape) < keep) / keep  
        return x * self.mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        if self.mask is None:
            return grad_out
        return grad_out * self.mask
