"""Run without Kafka or PostgreSQL; verify .npz hot-reload contract."""
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np


def test_hot_reload_without_restarting_process(tmp_path, monkeypatch):
    fake_preprocessing = types.ModuleType("preprocessing")
    fake_preprocessing.FEATURE_NAMES = ["x", "y"]
    fake_preprocessing.to_feature_vector = lambda features: np.asarray(
        [features["x"], features["y"]], dtype=float
    )
    monkeypatch.setitem(sys.modules, "preprocessing", fake_preprocessing)
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps({
        "feature_names": ["x", "y"], "mean": [0, 0], "scale": [1, 1],
        "coef": [2, 0], "intercept": 0, "threshold": 0.5,
    }))
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    monkeypatch.setenv("MODEL_PATH", str(baseline_path))
    scorer_path = Path(__file__).resolve().parents[1] / "scoring_service" / "scorer.py"
    spec = importlib.util.spec_from_file_location("hw3_test_scorer", scorer_path)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)

    score1, _, ver1 = scorer.score_transaction({"x": 1, "y": 0})
    assert ver1 == "baseline_v1" and score1 > .5

    np.savez_compressed(model_dir / "model_v2.npz",
        feature_names=np.array(["x", "y"]), mean=np.array([0., 0.]),
        scale=np.array([1., 1.]), coef=np.array([-2., 0.]),
        intercept=np.array(0.), threshold=np.array(.5))
    (model_dir / "latest.txt").write_text("v2\n")
    score2, _, ver2 = scorer.score_transaction({"x": 1, "y": 0})
    assert ver2 == "v2" and score2 < .5
    assert score2 != score1

    (model_dir / "latest.txt").write_text("v3\n")
    score3, _, ver3 = scorer.score_transaction({"x": 1, "y": 0})
    assert ver3 == "v2" and score3 == score2  # missing binary => last-good model
