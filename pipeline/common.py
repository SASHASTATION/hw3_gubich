"""Shared DB, month arithmetic and atomic file helpers for Airflow tasks."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import psycopg2
from dateutil.relativedelta import relativedelta

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts"))
MODEL_DIR = ARTIFACTS_DIR / "models"
REPORT_DIR = ARTIFACTS_DIR / "reports"
WORK_DIR = ARTIFACTS_DIR / "work"
WINDOW_MONTHS = int(os.environ.get("WINDOW_MONTHS", "3"))
HISTORY_MONTHS = int(os.environ.get("HISTORY_MONTHS", "36"))
SUPPLY_BATCH_SIZE = int(os.environ.get("SUPPLY_BATCH_SIZE", "1000"))


def month_start(value: date) -> date:
    """Get first day of calendar month (uses date, not an approximate 30-day step)."""
    return value.replace(day=1)


def add_months(value: date, count: int) -> date:
    """Advance a date by count calendar months, preserving day where possible."""
    return value + relativedelta(months=count)


@contextmanager
def connect():
    """Yield transaction-scoped fraud_db connection; commit only on success."""
    connection = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "postgres"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "fraud_db"),
        user=os.environ.get("POSTGRES_USER", "fraud_user"),
        password=os.environ.get("POSTGRES_PASSWORD", "fraud_password"),
    )
    try:
        yield connection
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def airflow_run_key() -> str:
    """Stable Airflow run_id across retries, used as SQL idempotency key."""
    from airflow.operators.python import get_current_context

    context = get_current_context()
    return f"{context['dag'].dag_id}/{context['run_id']}"


def work_dir(run_key: str) -> Path:
    """Stable directory for all attempts of one DAG run."""
    directory = WORK_DIR / hashlib.sha256(run_key.encode()).hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def atomic_write_text(path: Path, content: str) -> None:
    """Replace content atomically: readers never observe half-written artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as file:
            tmp_name = file.name
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_json(path: Path, obj: dict) -> None:
    """Write JSON artifact via atomic rename."""
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def model_path(version: str) -> Path:
    """Return artifact path for version; restrict string to safe filename chars."""
    if not version or not all(ch.isalnum() or ch in "_-" for ch in version):
        raise ValueError(f"Unsafe model version {version!r}")
    return MODEL_DIR / f"model_{version}.npz"


def atomic_save_npz(path: Path, fields: dict) -> None:
    """Atomically save a CPU model as a compressed NumPy binary (allow_pickle=False)."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as file:
            tmp_name = file.name
            np.savez_compressed(file, **fields)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def read_npz_model(path: Path) -> dict:
    """Read a published NumPy binary into the same mathematical model schema."""
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        return {
            "feature_names": data["feature_names"].tolist(),
            "mean": data["mean"].tolist(),
            "scale": data["scale"].tolist(),
            "coef": data["coef"].tolist(),
            "intercept": float(data["intercept"]),
            "threshold": float(data["threshold"]),
        }
