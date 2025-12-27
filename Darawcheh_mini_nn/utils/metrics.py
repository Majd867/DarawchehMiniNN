from __future__ import annotations
import numpy as np
from typing import Iterator, Tuple


def batch_iter(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)

    for start in range(0, N, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]
