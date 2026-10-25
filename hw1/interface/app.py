import json
import os
import time
import urllib.request
import uuid

import numpy as np
import pandas as pd
import psycopg2
import streamlit as st
from confluent_kafka import Producer


KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TRANSACTIONS_TOPIC = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
SCORING_HEALTH_URL = os.getenv("SCORING_HEALTH_URL", "http://scoring-service:8001/health")
WRITER_HEALTH_URL = os.getenv("WRITER_HEALTH_URL", "http://results-writer:8002/health")

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname": os.getenv("POSTGRES_DB", "fraud_db"),
    "user": os.getenv("POSTGRES_USER", "fraud_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "fraud_password"),
}


def json_safe_record(row: pd.Series) -> dict:
    result = {}
    for key, value in row.to_dict().items():
        if pd.isna(value):
            result[key] = None
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def transaction_id_from_row(record: dict) -> str:
    for field in ("transaction_id", "id", "ID", "trans_num"):
        value = record.get(field)
        if value is not None and not pd.isna(value):
            return str(value)
    return str(uuid.uuid4())


def send_transactions(dataframe: pd.DataFrame, limit: int) -> int:
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})
    rows = dataframe.head(limit)
    progress = st.progress(0)

    for position, (_, row) in enumerate(rows.iterrows(), start=1):
        record = json_safe_record(row)
        transaction_id = transaction_id_from_row(record)
        payload = {"transaction_id": transaction_id, "data": record}

        producer.produce(
            TRANSACTIONS_TOPIC,
            key=transaction_id.encode("utf-8"),
            value=json.dumps(payload, default=str).encode("utf-8"),
        )
        producer.poll(0)
        progress.progress(position / len(rows))

    producer.flush(10)
    return len(rows)


def db_connection():
    return psycopg2.connect(**DB_CONFIG)


def load_dashboard_summary() -> dict:
    query = """
        SELECT
            COUNT(*) AS total_transactions,
            COUNT(*) FILTER (WHERE fraud_flag = 1) AS fraud_transactions,
            COALESCE(AVG(score), 0) AS avg_score
        FROM transaction_scores
    """

    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            total, fraud, avg_score = cursor.fetchone()

    fraud_rate = (fraud / total * 100.0) if total else 0.0
    return {
        "total": int(total),
        "fraud": int(fraud),
        "fraud_rate": float(fraud_rate),
        "avg_score": float(avg_score),
    }


def query_dataframe(query: str) -> pd.DataFrame:
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [column.name for column in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def load_fraud_transactions() -> pd.DataFrame:
    return query_dataframe(
        """
        SELECT transaction_id, score, fraud_flag, created_at
        FROM transaction_scores
        WHERE fraud_flag = 1
        ORDER BY created_at DESC
        LIMIT 10
        """
    )


def load_recent_scores() -> pd.DataFrame:
    return query_dataframe(
        """
        SELECT transaction_id, score, fraud_flag, created_at
        FROM transaction_scores
        ORDER BY created_at DESC
        LIMIT 100
        """
    )


def render_histogram(scores: pd.Series):
    if scores.empty:
        st.info("В PostgreSQL пока нет результатов скоринга.")
        return

    bins_count = min(10, max(1, len(scores)))
    counts, bins = np.histogram(scores.astype(float), bins=bins_count, range=(0.0, 1.0))
    labels = [
        f"{bins[index]:.2f}–{bins[index + 1]:.2f}"
        for index in range(len(counts))
    ]
    histogram = pd.DataFrame(
        {"score interval": labels, "transactions": counts}
    ).set_index("score interval")
    st.bar_chart(histogram)


def check_kafka() -> tuple[bool, str]:
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})
    try:
        metadata = producer.list_topics(timeout=2)
        if metadata.brokers:
            return True, f"{len(metadata.brokers)} broker(s)"
        return False, "no brokers"
    except Exception as error:
        return False, str(error)


def check_postgres() -> tuple[bool, str]:
    try:
        with db_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
        return True, "connected"
    except Exception as error:
        return False, str(error)


def check_http_health(url: str) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status == 200, payload.get("status", "ok")
    except Exception as error:
        return False, str(error)


def render_status_card(name: str, status: tuple[bool, str]):
    healthy, detail = status
    icon = "🟢" if healthy else "🔴"
    st.metric(name, f"{icon} {'Healthy' if healthy else 'Unavailable'}")
    st.caption(detail[:80])


st.set_page_config(page_title="Fraud Scoring", page_icon="💳", layout="wide")
st.title("Real-Time Fraud Scoring Dashboard")
st.caption("Kafka → preprocessing → ML inference → scores → PostgreSQL")

send_tab, dashboard_tab = st.tabs(["Отправка транзакций", "Dashboard"])

with send_tab:
    st.subheader("Отправка test.csv в Kafka")
    uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])

    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        st.write(f"Строк в файле: **{len(dataframe)}**")
        st.dataframe(dataframe.head(10), use_container_width=True)

        default_limit = min(100, len(dataframe))
        limit = st.number_input(
            "Сколько транзакций отправить",
            min_value=1,
            max_value=max(1, len(dataframe)),
            value=max(1, default_limit),
            step=1,
        )

        if st.button("Отправить в Kafka", type="primary"):
            try:
                sent = send_transactions(dataframe, int(limit))
                st.success(f"Отправлено транзакций: {sent}")
            except Exception as error:
                st.error(f"Не удалось отправить данные: {error}")

with dashboard_tab:
    st.subheader("Monitoring")
    st.caption("Данные автоматически обновляются каждые 3 секунды.")

    @st.fragment(run_every="3s")
    def live_dashboard():
        try:
            summary = load_dashboard_summary()
            recent_scores_df = load_recent_scores()
            fraud_df = load_fraud_transactions()

            metric_cols = st.columns(4)
            metric_cols[0].metric("Transactions", summary["total"])
            metric_cols[1].metric("Fraud detected", summary["fraud"])
            metric_cols[2].metric("Fraud rate", f"{summary['fraud_rate']:.2f}%")
            metric_cols[3].metric("Average score", f"{summary['avg_score']:.3f}")

            st.markdown("#### Infrastructure status")
            status_cols = st.columns(4)
            statuses = [
                ("Kafka", check_kafka()),
                ("PostgreSQL", check_postgres()),
                ("Scoring service", check_http_health(SCORING_HEALTH_URL)),
                ("Results writer", check_http_health(WRITER_HEALTH_URL)),
            ]
            for column, (name, status) in zip(status_cols, statuses):
                with column:
                    render_status_card(name, status)

            left, right = st.columns([1, 1])
            with left:
                st.markdown("#### 10 последних fraud-транзакций")
                if fraud_df.empty:
                    st.info("Fraud-транзакций пока нет.")
                else:
                    st.dataframe(fraud_df, use_container_width=True, hide_index=True)

            with right:
                st.markdown("#### Score distribution — последние 100")
                render_histogram(recent_scores_df["score"] if not recent_scores_df.empty else pd.Series(dtype=float))

            if not recent_scores_df.empty:
                st.markdown("#### Последние результаты")
                st.dataframe(
                    recent_scores_df.head(20),
                    use_container_width=True,
                    hide_index=True,
                )

            st.caption(f"Last refresh: {time.strftime('%H:%M:%S')}")

        except Exception as error:
            st.error(f"Не удалось обновить dashboard: {error}")

    live_dashboard()
