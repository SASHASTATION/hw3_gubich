import json
import logging
import os
import signal
import time

import psycopg2
from confluent_kafka import Consumer, KafkaError

from health import start_health_server


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("results-writer")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
SCORES_TOPIC = os.getenv("KAFKA_SCORES_TOPIC", "scores")
HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8002"))

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname": os.getenv("POSTGRES_DB", "fraud_db"),
    "user": os.getenv("POSTGRES_USER", "fraud_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "fraud_password"),
}

running = True


def stop_service(*_):
    global running
    running = False


def connect_postgres():
    while running:
        try:
            connection = psycopg2.connect(**DB_CONFIG)
            connection.autocommit = True
            logger.info("Connected to PostgreSQL")
            return connection
        except psycopg2.OperationalError as error:
            logger.warning("PostgreSQL is not ready: %s", error)
            time.sleep(2)

    raise RuntimeError("Service stopped before PostgreSQL connection was established")


def save_score(connection, result: dict):
    query = """
        INSERT INTO transaction_scores (transaction_id, score, fraud_flag)
        VALUES (%s, %s, %s)
        ON CONFLICT (transaction_id)
        DO UPDATE SET
            score = EXCLUDED.score,
            fraud_flag = EXCLUDED.fraud_flag,
            created_at = NOW()
    """

    with connection.cursor() as cursor:
        cursor.execute(
            query,
            (
                str(result["transaction_id"]),
                float(result["score"]),
                int(result["fraud_flag"]),
            ),
        )


def main():
    connection = connect_postgres()
    health_server = start_health_server(HEALTH_PORT, "results-writer")
    logger.info("Health endpoint started on port %d", HEALTH_PORT)

    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "group.id": "fraud-results-writer-v1",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([SCORES_TOPIC])

    logger.info("Listening to Kafka topic '%s'", SCORES_TOPIC)

    try:
        while running:
            message = consumer.poll(1.0)

            if message is None:
                continue

            if message.error():
                if message.error().code() != KafkaError._PARTITION_EOF:
                    logger.error("Kafka error: %s", message.error())
                continue

            try:
                result = json.loads(message.value().decode("utf-8"))
                save_score(connection, result)
                consumer.commit(message=message, asynchronous=False)

                logger.info("Saved %s to PostgreSQL", result["transaction_id"])

            except (psycopg2.InterfaceError, psycopg2.OperationalError):
                logger.warning("PostgreSQL connection was lost; reconnecting")
                connection = connect_postgres()

                try:
                    save_score(connection, result)
                    consumer.commit(message=message, asynchronous=False)
                    logger.info(
                        "Saved %s to PostgreSQL after reconnect",
                        result["transaction_id"],
                    )
                except Exception:
                    logger.exception("Failed to persist score after reconnect")

            except Exception:
                logger.exception("Failed to persist score message")

    finally:
        consumer.close()
        connection.close()
        health_server.shutdown()
        logger.info("Results writer stopped")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop_service)
    signal.signal(signal.SIGINT, stop_service)
    main()
