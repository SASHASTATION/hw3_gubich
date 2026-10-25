"""Unit checks can run on host: pip install pytest numpy scikit-learn."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.metrics import evaluate_model, predict_scores


def test_json_logistic_predicts_probability():
    model = {
        "mean": [0.0, 0.0], "scale": [1.0, 1.0],
        "coef": [3.0, 0.0], "intercept": 0.0, "threshold": 0.5,
    }
    xs = np.asarray([[-2.0, 10], [2.0, -5]])
    scores = predict_scores(model, xs)
    assert scores[0] < 0.01 and scores[1] > 0.99
    metrics = evaluate_model(model, xs, np.array([0, 1]))
    assert metrics["average_precision"] == 1.0
    assert metrics["f1"] == 1.0
