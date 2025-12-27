from __future__ import annotations
from typing import List
import numpy as np
from .base import Optimizer
from ..core.parameter import Parameter


class Adam(Optimizer):
    def __init__(self, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = {}
        self.v = {}
        self.t = 0



    def step(self, params: List[Parameter]) -> None:
        self.t += 1
        for p in params:
            pid = id(p)
            if pid not in self.m:
                self.m[pid] = np.zeros_like(p.data)
                self.v[pid] = np.zeros_like(p.data)

            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            self.m[pid] = self.b1 * self.m[pid] + (1 - self.b1) * grad
            self.v[pid] = self.b2 * self.v[pid] + (1 - self.b2) * (grad * grad)

            m_hat = self.m[pid] / (1 - self.b1 ** self.t)
            v_hat = self.v[pid] / (1 - self.b2 ** self.t)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
