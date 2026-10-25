"""Every five minutes train a versioned model, compare, then conditionally publish."""
from datetime import datetime, timedelta, timezone

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

from pipeline.retrain import (
    claim_training_month, choose_promotion, evaluate_candidate, fit_candidate,
    prepare_dataset, publish_model, reject_model,
)

with DAG(
    dag_id="retrain_fraud_model",
    description="Temporal rolling-window training with incumbent AP promotion gate",
    schedule="*/5 * * * *",
    start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "mlops", "retries": 2, "retry_delay": timedelta(seconds=20)},
    tags=["hw3", "training"],
) as dag:
    claim = PythonOperator(task_id="claim_training_month", python_callable=claim_training_month)
    prepare = PythonOperator(task_id="prepare_dataset", python_callable=prepare_dataset)
    fit = PythonOperator(task_id="fit_candidate", python_callable=fit_candidate)
    evaluate = PythonOperator(task_id="evaluate_candidate", python_callable=evaluate_candidate)
    decide = BranchPythonOperator(task_id="decide_promotion", python_callable=choose_promotion)
    publish = PythonOperator(task_id="publish_model", python_callable=publish_model)
    reject = PythonOperator(task_id="reject_model", python_callable=reject_model)
    claim >> prepare >> fit >> evaluate >> decide >> [publish, reject]
