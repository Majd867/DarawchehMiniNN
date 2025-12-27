from __future__ import annotations
from typing import List
from .base import Optimizer
from ..core.parameter import Parameter
class SGD(Optimizer):


    def __init__(self, lr: float = 1e-2, weight_decay: float = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay


    def step(self, params: List[Parameter]) -> None:
        for p in params:
            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.data
            p.data -= self.lr * grad
