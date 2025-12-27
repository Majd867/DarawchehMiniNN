from __future__ import annotations
import numpy as np
from ..core.layer import Layer

class Linear(Layer):
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        return x


    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out




class ReLU(Layer):
    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self.mask is not None
        return grad_out * self.mask







class Sigmoid(Layer):
    def __init__(self):
        self.out = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        out = 1.0 / (1.0 + np.exp(-x))
        self.out = out
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self.out is not None
        return grad_out * self.out * (1.0 - self.out)




class Tanh(Layer):
    def __init__(self):
        self.out = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self.out is not None
        return grad_out * (1.0 - self.out**2)
