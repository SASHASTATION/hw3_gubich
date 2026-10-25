"""Separate, retry-safe tasks: claim -> prepare -> fit -> evaluate -> publish."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date

import numpy as np
from airflow.exceptions import AirflowSkipException
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pipeline.common import (
    MODEL_DIR, WINDOW_MONTHS, add_months, airflow_run_key, atomic_json,
    atomic_write_text, atomic_save_npz, read_npz_model, connect, model_path, work_dir,
)
from pipeline.metrics import evaluate_model

# Reuse EXACT online feature extraction (not a copy of seminar preprocessing).
from preprocessing import FEATURE_NAMES, preprocess_transaction, to_feature_vector


def _run_target(cur, key):
    cur.execute("SELECT target_month FROM training_runs WHERE run_key=%s", (key,))
    row = cur.fetchone()
    if not row:
        raise RuntimeError("claim_training_month must succeed first")
    return row[0]


def claim_training_month() -> None:
    """Read supply/training cursors; bind target month to stable run ID.

    If retried after successful publish, skip without shifting month pointer.
    Claiming before work makes retries keep the same candidate window.
    """
    key = airflow_run_key()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT last_model_month FROM training_state WHERE id=1 FOR UPDATE")
            last_model = cur.fetchone()[0]
            cur.execute("SELECT target_month FROM training_runs WHERE run_key=%s", (key,))
            existing = cur.fetchone()
            if existing:
                target = existing[0]
            else:
                cur.execute("SELECT last_month FROM supply_state WHERE id=1")
                supplied = cur.fetchone()
                if not supplied:
                    raise ValueError("No labeled months; bootstrap not complete")
                target = supplied[0]
                if last_model is not None and target <= last_model:
                    raise AirflowSkipException("No newly supplied month to train")
                cur.execute(
                    "INSERT INTO training_runs(run_key,target_month) VALUES (%s,%s)",
                    (key, target),
                )
    if last_model is not None and target <= last_model:
        raise AirflowSkipException(f"Month {target} already deployed")
    print(f"Retrain target month: {target}")


def prepare_dataset() -> None:
    """Read latest WINDOW_MONTHS raw labeled months; write deterministic NPZ.

    Older WINDOW_MONTHS-1 months are train, newest month is validation;
    rerunning overwrites only this run's local snapshot, not production data.
    """
    key = airflow_run_key()
    with connect() as conn:
        with conn.cursor() as cur:
            target = _run_target(cur, key)
            start = add_months(target, -(WINDOW_MONTHS - 1))
            cur.execute(
                """SELECT batch_month, payload, is_fraud FROM labeled_transactions
                   WHERE batch_month BETWEEN %s AND %s
                   ORDER BY transaction_time, id""", (start, target)
            )
            rows = cur.fetchall()
    months = {add_months(start, i) for i in range(WINDOW_MONTHS)}
    actual_months = {month for month, _, _ in rows}
    if months != actual_months:
        raise ValueError(f"Window missing months: expected {months}, got {actual_months}")
    if WINDOW_MONTHS < 2:
        raise ValueError("WINDOW_MONTHS must be at least 2")
    x_train, y_train, x_test, y_test = [], [], [], []
    for month, payload, label in rows:
        vector = to_feature_vector(preprocess_transaction(payload))
        if not np.isfinite(vector).all():
            raise ValueError("Non-finite features in labeled showcase")
        if month == target:
            x_test.append(vector)
            y_test.append(int(label))
        else:
            x_train.append(vector)
            y_train.append(int(label))
    if len(set(y_train)) != 2 or len(set(y_test)) != 2:
        raise ValueError("Both classes must be present in train and validation")
    output = work_dir(key) / "split.npz"
    # Writing to stable run path is fine for retries; the same train window is frozen by month.
    np.savez_compressed(
        output, x_train=np.asarray(x_train), y_train=np.asarray(y_train),
        x_test=np.asarray(x_test), y_test=np.asarray(y_test)
    )
    atomic_json(work_dir(key) / "window.json", {
        "start_month": str(start), "test_month": str(target),
        "train_months": WINDOW_MONTHS - 1, "train_rows": len(y_train), "test_rows": len(y_test),
    })
    print(f"Train rows={len(y_train)}; test rows={len(y_test)}, test month={target}")


def fit_candidate() -> None:
    """Read split.npz; fit CPU logistic regression; write candidate.json.

    Every attempt for one Airflow run overwrites the same private candidate.
    Production model directory is never modified by this task.
    """
    key = airflow_run_key()
    with np.load(work_dir(key) / "split.npz") as data:
        x_train, y_train = data["x_train"], data["y_train"]
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    estimator = LogisticRegression(
        C=1.0, class_weight="balanced", solver="lbfgs", max_iter=400, random_state=42
    )
    estimator.fit(x_scaled, y_train)
    candidate = {
        "model_type": "logistic_regression",
        "feature_names": FEATURE_NAMES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": estimator.coef_[0].tolist(),
        "intercept": float(estimator.intercept_[0]),
        "threshold": 0.5,
    }
    atomic_json(work_dir(key) / "candidate.json", candidate)
    print("Candidate trained and saved for evaluation")


def evaluate_candidate() -> None:
    """Read candidate/incumbent and SAME held-out month; write metrics/decision.

    Never touches prod. The promotion criterion is average precision >= incumbent
    average precision on identical temporal holdout (no random split).
    """
    key = airflow_run_key()
    directory = work_dir(key)
    with open(directory / "candidate.json", encoding="utf-8") as file:
        candidate = json.load(file)
    with np.load(directory / "split.npz") as data:
        x_test, y_test = data["x_test"], data["y_test"]
    candidate_metrics = evaluate_model(candidate, x_test, y_test)
    pointer = MODEL_DIR / "latest.txt"
    incumbent_metrics = None
    if pointer.exists():
        previous_version = pointer.read_text(encoding="utf-8").strip()
        incumbent = read_npz_model(model_path(previous_version))
        incumbent_metrics = evaluate_model(incumbent, x_test, y_test)
    # Initial publication has no Airflow-published incumbent; baseline remains fallback.
    approved = (incumbent_metrics is None or
                candidate_metrics["average_precision"] + 1e-12 >=
                incumbent_metrics["average_precision"])
    decision = {
        "candidate": candidate_metrics,
        "incumbent": incumbent_metrics,
        "approved": bool(approved),
        "promotion_metric": "average_precision",
    }
    atomic_json(directory / "decision.json", decision)
    print(f"Promotion decision: {decision}")


def choose_promotion() -> str:
    """Read decision.json; branch; retries use deterministic report, no DB writes."""
    with open(work_dir(airflow_run_key()) / "decision.json", encoding="utf-8") as file:
        decision = json.load(file)
    return "publish_model" if decision["approved"] else "reject_model"


def reject_model() -> None:
    """Read metrics; persist rejection history but never advance training pointer."""
    key = airflow_run_key()
    with open(work_dir(key) / "decision.json", encoding="utf-8") as file:
        decision = json.load(file)
    with connect() as conn:
        with conn.cursor() as cur:
            target = _run_target(cur, key)
            version = f"m{target:%Y%m}_{hashlib.sha256(key.encode()).hexdigest()[:10]}"
            cur.execute(
                """INSERT INTO model_history(version,target_month,candidate_ap,incumbent_ap,approved)
                   VALUES(%s,%s,%s,%s,false) ON CONFLICT (version) DO NOTHING""",
                (version, target, decision["candidate"]["average_precision"],
                 decision["incumbent"]["average_precision"])
            )
            cur.execute("UPDATE training_runs SET status='rejected' WHERE run_key=%s", (key,))
    print("Candidate worse than active model: NOT promoted; training cursor unchanged")


def publish_model() -> None:
    """Read approved candidate; atomically publish model then advance DB cursor.

    Reuse deterministic version per run on retries. The production pointer is
    switched only after the complete binary model file exists; DB cursor updated LAST.
    """
    key = airflow_run_key()
    directory = work_dir(key)
    candidate = json.loads((directory / "candidate.json").read_text(encoding="utf-8"))
    decision = json.loads((directory / "decision.json").read_text(encoding="utf-8"))
    if not decision["approved"]:
        raise ValueError("Refusing to publish a rejected candidate")
    with connect() as conn:
        with conn.cursor() as cur:
            target = _run_target(cur, key)
    version = f"m{target:%Y%m}_{hashlib.sha256(key.encode()).hexdigest()[:10]}"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    atomic_save_npz(model_path(version), {
        "feature_names": np.asarray(candidate["feature_names"], dtype=str),
        "mean": np.asarray(candidate["mean"], dtype=float),
        "scale": np.asarray(candidate["scale"], dtype=float),
        "coef": np.asarray(candidate["coef"], dtype=float),
        "intercept": np.asarray(candidate["intercept"], dtype=float),
        "threshold": np.asarray(candidate["threshold"], dtype=float),
    })
    # `latest.txt` is a single atomic pointer used by the live scorer.
    atomic_write_text(MODEL_DIR / "latest.txt", version + "\n")
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO model_history(version,target_month,candidate_ap,incumbent_ap,approved)
                   VALUES(%s,%s,%s,%s,true) ON CONFLICT (version) DO NOTHING""",
                (version, target, decision["candidate"]["average_precision"],
                 (decision["incumbent"] or {}).get("average_precision"))
            )
            cur.execute(
                """UPDATE training_state
                   SET last_model_month=%s, active_version=%s WHERE id=1
                   AND (last_model_month IS NULL OR last_model_month<=%s)""",
                (target, version, target)
            )
            cur.execute("UPDATE training_runs SET status='published' WHERE run_key=%s", (key,))
    print(f"Published {model_path(version)}; switched latest.txt to {version}")
