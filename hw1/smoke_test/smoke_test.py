import json
import os
import sys
import time
import uuid

import psycopg2
from confluent_kafka import Producer


KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TRANSACTIONS_TOPIC = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "postgres"),
    "port": int(os.getenv("POSTGRES_PORT", "5432")),
    "dbname": os.getenv("POSTGRES_DB", "fraud_db"),
    "user": os.getenv("POSTGRES_USER", "fraud_user"),
    "password": os.getenv("POSTGRES_PASSWORD", "fraud_password"),
}


def ok(message):
    print(f"✓ {message}")


def fail(message):
    print(f"✗ {message}")
    sys.exit(1)


def main():
    transaction_id = f"smoke-{uuid.uuid4().hex[:10]}"

    print("\n=== Fraud scoring integration smoke test ===\n")

    try:
        producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})
        metadata = producer.list_topics(timeout=5)
        if not metadata.brokers:
            fail("Kafka has no available brokers")
        ok("Kafka is available")
    except Exception as error:
        fail(f"Kafka check failed: {error}")

    try:
        connection = psycopg2.connect(**DB_CONFIG)
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        ok("PostgreSQL is available")
    except Exception as error:
        fail(f"PostgreSQL check failed: {error}")

    payload = {
        "transaction_id": transaction_id,
        "data": {
            "transaction_id": transaction_id,
            "transaction_time": "2025-05-03 02:15:00",
            "amount": 4500.0,
            "lat": 40.7128,
            "lon": -74.0060,
            "merchant_lat": 34.0522,
            "merchant_lon": -118.2437,
            "population_city": 8804190,
            "dob": "1998-11-03",
        },
    }

    try:
        producer.produce(
            TRANSACTIONS_TOPIC,
            key=transaction_id.encode("utf-8"),
            value=json.dumps(payload).encode("utf-8"),
        )
        remaining = producer.flush(10)
        if remaining != 0:
            fail("Kafka producer did not flush all messages")
        ok(f"Test transaction sent: {transaction_id}")
    except Exception as error:
        fail(f"Could not send transaction: {error}")

    deadline = time.time() + 30
    result = None

    while time.time() < deadline:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT transaction_id, score, fraud_flag
                FROM transaction_scores
                WHERE transaction_id = %s
                """,
                (transaction_id,),
            )
            result = cursor.fetchone()

        if result:
            break
        time.sleep(1)

    connection.close()

    if result is None:
        fail("No PostgreSQL result after 30 seconds")

    _, score, fraud_flag = result
    ok("Scoring result persisted to PostgreSQL")

    print("\nResult:")
    print(f"  transaction_id: {transaction_id}")
    print(f"  score:          {float(score):.6f}")
    print(f"  fraud_flag:     {int(fraud_flag)}")
    print("\nALL TESTS PASSED\n")


if __name__ == "__main__":
    main()
