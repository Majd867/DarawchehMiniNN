from __future__ import annotations
from typing import List
import numpy as np
from .base import Optimizer
from ..core.parameter import Parameter

class Momentum(Optimizer):
    def __init__(self, lr: float = 1e-2, momentum: float = 0.9, weight_decay: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = {}  


    def step(self, params: List[Parameter]) -> None:
        for p in params:
            pid = id(p)
            if pid not in self.v:
                self.v[pid] = np.zeros_like(p.data)
            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            self.v[pid] = self.momentum * self.v[pid] - self.lr * grad
            p.data += self.v[pid]
