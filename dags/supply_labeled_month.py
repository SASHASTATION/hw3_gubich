"""Every 10 minutes extend demo labeled showcase by one calendar month."""
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator

from pipeline.supply import append_month, claim_month, finalize_month

with DAG(
    dag_id="supply_labeled_month",
    description="Idempotently append a labeled month, then prune old history",
    schedule="*/10 * * * *",
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "mlops", "retries": 2, "retry_delay": timedelta(seconds=20)},
    tags=["hw3", "supply"],
) as dag:
    claim = PythonOperator(task_id="claim_month", python_callable=claim_month)
    append = PythonOperator(task_id="append_month", python_callable=append_month)
    finalize = PythonOperator(task_id="finalize_month", python_callable=finalize_month)
    claim >> append >> finalize
