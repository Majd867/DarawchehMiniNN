from __future__ import annotations
from typing import Callable, Dict, List, Tuple, Any
import numpy as np

class HyperparameterTuner:

    def __init__(self, build_fn: Callable[[Dict[str, Any]], tuple]):
        self.build_fn = build_fn


    def grid_search(
        self,
        configs:List[Dict[str, Any]],
        x_train:np.ndarray, y_train: np.ndarray,
        x_val:np.ndarray, y_val: np.ndarray
    ) -> Tuple[Tuple[float, Dict[str, Any]],List[Tuple[float, Dict[str, Any]]]]:


        best = (-1.0, {})
        results: List[Tuple[float, Dict[str, Any]]] = []

        for cfg in configs:
            model, trainer = self.build_fn(cfg)
            hist = trainer.fit(x_train, y_train, x_val, y_val, verbose=False)
            score = hist["val_acc"][-1] if hist["val_acc"] else model.accuracy(x_val, y_val)
            results.append((score, cfg))
            if score > best[0]:
                best = (score, cfg)



        results.sort(key=lambda x: x[0], reverse=True)
        return best, results
