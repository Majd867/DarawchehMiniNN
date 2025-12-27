from __future__ import annotations
import numpy as np
class Parameter:
   
    def __init__(self, data: np.ndarray):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)



    def zero_grad(self) -> None:
        self.grad[...] = 0.0
