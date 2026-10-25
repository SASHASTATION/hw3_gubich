"""CPU logistic JSON scorer; hot-reload when Airflow switches latest.txt.

Original HW1 JSON schema is still supported, so the original bundled baseline
model is used before the first successful Airflow training run.
"""
import json
import logging
import math
import os
import zipfile
from pathlib import Path

import numpy as np

from preprocessing import FEATURE_NAMES, to_feature_vector

logger = logging.getLogger("scoring-service")


class LogisticModel:
    """Load preprocessed coefficients, scaling parameters and threshold from JSON."""

    def __init__(self, model_path: str):
        if Path(model_path).suffix == ".npz":
            # Airflow publishes binary NumPy arrays; no scikit-learn dependency online.
            with np.load(model_path, allow_pickle=False) as data:
                artifact = {
                    "feature_names": data["feature_names"].tolist(),
                    "mean": data["mean"].tolist(),
                    "scale": data["scale"].tolist(),
                    "coef": data["coef"].tolist(),
                    "intercept": float(data["intercept"]),
                    "threshold": float(data["threshold"]),
                }
        else:
            with open(model_path, "r", encoding="utf-8") as file:
                artifact = json.load(file)
        if artifact["feature_names"] != FEATURE_NAMES:
            raise ValueError("Model feature order does not match preprocessing.py")
        self.mean = np.asarray(artifact["mean"], dtype=float)
        self.scale = np.asarray(artifact["scale"], dtype=float)
        self.coef = np.asarray(artifact["coef"], dtype=float)
        self.intercept = float(artifact["intercept"])
        self.threshold = float(artifact["threshold"])
        if not (len(self.mean) == len(self.scale) == len(self.coef) == len(FEATURE_NAMES)):
            raise ValueError("Wrong model array dimensions")

    def predict(self, features: dict) -> tuple[float, int]:
        x = to_feature_vector(features)
        safe_scale = np.where(self.scale == 0, 1.0, self.scale)
        logit = self.intercept + float(np.dot((x - self.mean) / safe_scale, self.coef))
        logit = max(-40.0, min(40.0, logit))
        score = 1.0 / (1.0 + math.exp(-logit))
        return score, int(score >= self.threshold)


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "model.json"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))

_active_version = "baseline_v1"
_active_model = None


def _load_current() -> tuple[LogisticModel, str]:
    """Read pointer on every message and load a new model only when version changes."""
    global _active_model, _active_version
    if _active_model is None:
        _active_model = LogisticModel(str(MODEL_PATH))
        _active_version = "baseline_v1"

    latest_pointer = MODEL_DIR / "latest.txt"
    if not latest_pointer.is_file():
        return _active_model, _active_version

    try:
        latest_version = latest_pointer.read_text(encoding="utf-8").strip()
        if not latest_version or not all(c.isalnum() or c in "_-" for c in latest_version):
            raise ValueError(f"Invalid model version: {latest_version!r}")
        if latest_version != _active_version:
            new_path = MODEL_DIR / f"model_{latest_version}.npz"
            new_model = LogisticModel(str(new_path))
            _active_model = new_model
            _active_version = latest_version
            logger.info("Hot-reloaded model version %s", latest_version)
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile) as error:
        logger.error("Model hot-reload failed; keeping active %s: %s", _active_version, error)
    return _active_model, _active_version


def score_transaction(features: dict) -> tuple[float, int, str]:
    """Return HW1 score and fraud flag plus the immutable model version."""
    model, version = _load_current()
    score, flag = model.predict(features)
    return score, flag, version
