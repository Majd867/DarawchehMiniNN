from __future__ import annotations
from typing import List
import numpy as np

from .base import Optimizer
from ..core.parameter import Parameter

class AdaGrad(Optimizer):


    def __init__(self, lr: float = 1e-2, eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay
        self.h = {} 



    def step(self, params: List[Parameter]) -> None:
        for p in params:
            pid = id(p)
            if pid not in self.h:
                self.h[pid] = np.zeros_like(p.data)

            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data

            self.h[pid] += grad * grad
            p.data -= self.lr * grad / (np.sqrt(self.h[pid]) + self.eps)
