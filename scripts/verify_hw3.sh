#!/usr/bin/env bash
set -euo pipefail

printf '\n### Airflow DAGs\n'
docker compose exec -T airflow-scheduler airflow dags list | grep -E 'supply_labeled_month|retrain_fraud_model|check_data_quality'
printf '\n### Published model\n'
if [ -f artifacts/models/latest.txt ]; then
  VERSION=$(cat artifacts/models/latest.txt)
  echo "active model: $VERSION"
  test -s "artifacts/models/model_${VERSION}.npz"
else
  echo 'No published model yet. Trigger retrain_fraud_model in UI or CLI.'
fi
printf '\n### Showcase and model state\n'
docker compose exec -T postgres psql -U fraud_user -d fraud_db -c "SELECT batch_month, count(*) AS rows FROM labeled_transactions GROUP BY 1 ORDER BY 1 DESC LIMIT 8"
docker compose exec -T postgres psql -U fraud_user -d fraud_db -c "SELECT * FROM training_state"
printf '\n### Recent scores with model version\n'
docker compose exec -T postgres psql -U fraud_user -d fraud_db -c "SELECT transaction_id, score, fraud_flag, model_version, created_at FROM transaction_scores ORDER BY created_at DESC LIMIT 5"
printf '\n### HTML reports\n'
ls -lh artifacts/reports/*.html 2>/dev/null || true
