from __future__ import annotations
import numpy as np
from ..core.network import Loss

class SoftmaxCrossEntropy(Loss):
    
    def __init__(self):
        self.probs = None
        self.y_true = None

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        self.y_true = y_true

        z = logits - np.max(logits, axis=1, keepdims=True)  
        expz = np.exp(z)
        probs = expz / np.sum(expz, axis=1, keepdims=True)
        self.probs = probs

        N = logits.shape[0]
        if y_true.ndim == 1:
            correct = probs[np.arange(N), y_true]
        else:
            correct = np.sum(probs * y_true, axis=1)

        return float(-np.mean(np.log(correct + 1e-12)))

    def backward(self) -> np.ndarray:
        assert self.probs is not None and self.y_true is not None
        N = self.probs.shape[0]
        grad = self.probs.copy()
        if self.y_true.ndim == 1:
            grad[np.arange(N), self.y_true] -= 1.0
        else:
            grad -= self.y_true
        return grad / N
