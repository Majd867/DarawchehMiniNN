from __future__ import annotations
from typing import List
from ..core.parameter import Parameter
class Optimizer:
    def step(self , params: List[Parameter],)  ->  None:
        raise NotImplementedError
