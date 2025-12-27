from __future__ import annotations
import numpy as np

def init_weights(in_features: int, out_features: int, mode: str = "xavier", seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if mode == "xavier":
        limit = np.sqrt(6.0 / (in_features + out_features))
        return rng.uniform(-limit, limit, size=(in_features, out_features)).astype(float)
    if mode == "he":
        return rng.normal(0.0, np.sqrt(2.0 / in_features), size=(in_features, out_features)).astype(float)
    return rng.normal(0.0, 0.01, size=(in_features, out_features)).astype(float)
