from __future__ import annotations
from typing import List
import numpy as np
from .parameter import Parameter



class Layer:

    def forward(self, x: np.ndarray , training: bool = True,) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_out: np.ndarray )  -> np.ndarray:
        raise NotImplementedError


    def parameters(self) -> List[Parameter]:
        return []

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()
