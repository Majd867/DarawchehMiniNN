from __future__ import annotations
import numpy as np

from ..core.layer import Layer
from ..core.parameter import Parameter
from ..utils.init import init_weights



class Dense(Layer):
    def __init__(self, in_features: int, out_features: int, weight_init: str = "xavier", seed: int | None = None):
        W = init_weights(in_features, out_features, mode=weight_init, seed=seed)
        b = np.zeros((1, out_features), dtype=float)

        self.W = Parameter(W)
        self.b = Parameter(b)
        self.x_cache: np.ndarray | None= None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.x_cache = x
        return x @ self.W.data + self.b.data

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self.x_cache is not None
        x = self.x_cache

        self.W.grad = x.T @ grad_out
        self.b.grad = np.sum(grad_out, axis=0, keepdims=True)

        grad_x = grad_out @ self.W.data.T
        return grad_x

    def parameters(self):
        return [self.W, self.b]
