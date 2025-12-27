from __future__ import annotations
import numpy as np

from ..core.layer import Layer
from ..core.parameter import Parameter

class BatchNorm1D(Layer):
   
    def __init__(self, dim: int, momentum: float = 0.9, eps: float = 1e-5):
        self.gamma = Parameter(np.ones((1, dim), dtype=float))
        self.beta = Parameter(np.zeros((1, dim), dtype=float))

        self.running_mean = np.zeros((1, dim), dtype=float)
        self.running_var = np.ones((1, dim), dtype=float)

        self.momentum = momentum
        self.eps = eps

        self.x_centered = None
        self.std_inv = None
        self.x_norm = None



    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mu = np.mean(x, axis=0, keepdims=True)
            var = np.var(x, axis=0, keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            self.x_centered = x - mu
            self.std_inv = 1.0 / np.sqrt(var + self.eps)
            self.x_norm = self.x_centered * self.std_inv

            return self.gamma.data * self.x_norm + self.beta.data
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma.data * x_norm + self.beta.data

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        assert self.x_centered is not None and self.std_inv is not None and self.x_norm is not None
        N = grad_out.shape[0]


        self.beta.grad = np.sum(grad_out, axis=0, keepdims=True)
        self.gamma.grad = np.sum(grad_out * self.x_norm, axis=0, keepdims=True)

        dx_norm = grad_out * self.gamma.data
        dvar = np.sum(dx_norm * self.x_centered, axis=0, keepdims=True) * (-0.5) * (self.std_inv ** 3)
        dmu = np.sum(dx_norm * (-self.std_inv), axis=0, keepdims=True) + dvar * np.mean(-2.0 * self.x_centered, axis=0, keepdims=True)


        dx = (dx_norm * self.std_inv) + (dvar * 2.0 * self.x_centered / N) + (dmu / N)
        return dx


    def parameters(self):
        return [self.gamma, self.beta]
