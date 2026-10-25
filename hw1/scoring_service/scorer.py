import json
import math
import os
from pathlib import Path

import numpy as np

from preprocessing import FEATURE_NAMES, to_feature_vector


class LogisticModel:
    """Small CPU-only logistic regression inference model."""

    def __init__(self, model_path: str):
        with open(model_path, "r", encoding="utf-8") as file:
            artifact = json.load(file)

        if artifact["feature_names"] != FEATURE_NAMES:
            raise ValueError("Model feature order does not match preprocessing.py")

        self.mean = np.array(artifact["mean"], dtype=float)
        self.scale = np.array(artifact["scale"], dtype=float)
        self.coef = np.array(artifact["coef"], dtype=float)
        self.intercept = float(artifact["intercept"])
        self.threshold = float(artifact["threshold"])

    def predict(self, features: dict) -> tuple[float, int]:
        x = to_feature_vector(features)
        safe_scale = np.where(self.scale == 0, 1.0, self.scale)
        x_scaled = (x - self.mean) / safe_scale

        logit = self.intercept + float(np.dot(x_scaled, self.coef))
        logit = max(-40.0, min(40.0, logit))
        score = 1.0 / (1.0 + math.exp(-logit))
        fraud_flag = int(score >= self.threshold)

        return score, fraud_flag


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.json"
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
model = LogisticModel(MODEL_PATH)


def score_transaction(features: dict) -> tuple[float, int]:
    return model.predict(features)
