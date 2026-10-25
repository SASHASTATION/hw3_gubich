"""Every ten minutes validate the labeled showcase and write HTML report."""
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.python import PythonOperator

from pipeline.dq import generate_report

with DAG(
    dag_id="check_data_quality",
    description="Nine DQ rules; save HTML even if rules fail",
    schedule="*/10 * * * *",
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "mlops", "retries": 1, "retry_delay": timedelta(seconds=20)},
    tags=["hw3", "quality"],
) as dag:
    quality_report = PythonOperator(task_id="generate_report", python_callable=generate_report)
