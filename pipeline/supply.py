"""Idempotent monthly labeled showcase extension, separately callable Airflow tasks."""
from __future__ import annotations

from datetime import datetime

from psycopg2.extras import Json, execute_values

from pipeline.common import (
    HISTORY_MONTHS, SUPPLY_BATCH_SIZE, add_months, airflow_run_key, connect
)


def claim_month() -> None:
    """Read supply_state; store this run's source/target months in supply_batches.

    On retry, the same run_key always resolves to the original target even when
    supply_state was already advanced by a completed previous attempt.
    """
    key = airflow_run_key()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT last_month FROM supply_state WHERE id=1 FOR UPDATE")
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("Supply state missing; run airflow-init/bootstrap first")
            cur.execute(
                "SELECT source_month, target_month FROM supply_batches WHERE run_key=%s", (key,)
            )
            planned = cur.fetchone()
            if planned is None:
                source = row[0]
                target = add_months(source, 1)
                cur.execute(
                    """INSERT INTO supply_batches(run_key, source_month, target_month)
                       VALUES (%s,%s,%s)""", (key, source, target)
                )
            else:
                source, target = planned
    print(f"Supply run {key}: {source} -> {target}")


def append_month() -> None:
    """Read rows of claimed source month and insert shifted copies in target month.

    Unique (sample_key, batch_month) + ON CONFLICT makes all retries safe.
    transaction_time is shifted by a calendar month, with payload time updated.
    """
    key = airflow_run_key()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT source_month, target_month FROM supply_batches WHERE run_key=%s", (key,)
            )
            planned = cur.fetchone()
            if not planned:
                raise RuntimeError("claim_month must run first")
            source, target = planned
            cur.execute(
                """SELECT sample_key, transaction_time, payload, is_fraud
                   FROM labeled_transactions WHERE batch_month=%s
                   ORDER BY sample_key LIMIT %s""", (source, SUPPLY_BATCH_SIZE)
            )
            rows = cur.fetchall()
            if not rows:
                raise ValueError(f"Source month {source} contains no rows")
            shifted = []
            for sample_key, old_dt, payload, label in rows:
                new_dt = add_months(old_dt, 1)
                new_payload = dict(payload)
                new_payload["transaction_time"] = new_dt.isoformat(sep=" ")
                new_payload["trans_date_trans_time"] = new_dt.isoformat(sep=" ")
                if "unix_time" in new_payload:
                    new_payload["unix_time"] = int(new_dt.timestamp())
                shifted.append((sample_key, target, new_dt, Json(new_payload), int(label)))
            execute_values(
                cur,
                """INSERT INTO labeled_transactions
                     (sample_key, batch_month, transaction_time, payload, is_fraud)
                     VALUES %s ON CONFLICT (sample_key, batch_month) DO NOTHING""",
                shifted, page_size=500
            )
    print(f"Copied up to {len(shifted)} rows: {source} -> {target}")


def finalize_month() -> None:
    """Read claimed target, prune old showcase months, advance pointer in one commit.

    Retried runs re-use target from supply_batches and never advance twice. State
    is written last; a failed transaction cannot leave a half-updated pointer.
    """
    key = airflow_run_key()
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT target_month FROM supply_batches WHERE run_key=%s FOR UPDATE", (key,)
            )
            record = cur.fetchone()
            if not record:
                raise RuntimeError("No claimed month")
            target = record[0]
            cur.execute("SELECT COUNT(*) FROM labeled_transactions WHERE batch_month=%s", (target,))
            count = cur.fetchone()[0]
            if count == 0:
                raise RuntimeError("Target month is empty; keep supply cursor unchanged")
            cutoff = add_months(target, -(HISTORY_MONTHS - 1))
            cur.execute("DELETE FROM labeled_transactions WHERE batch_month < %s", (cutoff,))
            cur.execute(
                "UPDATE supply_state SET last_month=GREATEST(last_month,%s) WHERE id=1", (target,)
            )
            cur.execute(
                "UPDATE supply_batches SET status='completed' WHERE run_key=%s", (key,)
            )
    print(f"Month {target} committed with {count} rows; retained from {cutoff}")
