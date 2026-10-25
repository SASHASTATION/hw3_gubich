"""Pure probability scoring for the JSON logistic-regression artifact format."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score


def predict_scores(model: dict, x: np.ndarray) -> np.ndarray:
    """Apply saved scaling and weights, identical mathematically to online scorer."""
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    weights = np.asarray(model["coef"], dtype=float)
    scale = np.where(scale == 0, 1.0, scale)
    logits = ((x - mean) / scale) @ weights + float(model["intercept"])
    logits = np.clip(logits, -40, 40)
    return 1.0 / (1.0 + np.exp(-logits))


def evaluate_model(model: dict, x: np.ndarray, y: np.ndarray) -> dict:
    """Return AP (promotion metric) and F1 on fixed 0.5 fraud threshold."""
    scores = predict_scores(model, x)
    pred = (scores >= float(model["threshold"])).astype(int)
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
