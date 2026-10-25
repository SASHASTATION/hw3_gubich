import json
import logging
import os
import signal

from confluent_kafka import Consumer, KafkaError, Producer

from health import start_health_server
from preprocessing import preprocess_transaction
from scorer import score_transaction


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("scoring-service")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TRANSACTIONS_TOPIC = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
SCORES_TOPIC = os.getenv("KAFKA_SCORES_TOPIC", "scores")
HEALTH_PORT = int(os.getenv("HEALTH_PORT", "8001"))

running = True


def stop_service(*_):
    global running
    running = False


def delivery_report(error, message):
    if error is not None:
        logger.error("Kafka delivery failed: %s", error)


def main():
    health_server = start_health_server(HEALTH_PORT, "scoring-service")
    logger.info("Health endpoint started on port %d", HEALTH_PORT)

    consumer = Consumer(
        {
            "bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS,
            "group.id": "fraud-scoring-v1",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    producer = Producer({"bootstrap.servers": KAFKA_BOOTSTRAP_SERVERS})

    consumer.subscribe([TRANSACTIONS_TOPIC])
    logger.info("Listening to Kafka topic '%s'", TRANSACTIONS_TOPIC)

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
                payload = json.loads(message.value().decode("utf-8"))
                transaction_id = str(payload["transaction_id"])
                raw_transaction = payload["data"]

                features = preprocess_transaction(raw_transaction)
                score, fraud_flag = score_transaction(features)

                result = {
                    "transaction_id": transaction_id,
                    "score": round(float(score), 6),
                    "fraud_flag": int(fraud_flag),
                }

                producer.produce(
                    SCORES_TOPIC,
                    key=transaction_id.encode("utf-8"),
                    value=json.dumps(result).encode("utf-8"),
                    callback=delivery_report,
                )
                producer.flush()

                consumer.commit(message=message, asynchronous=False)
                logger.info(
                    "Scored %s: score=%.4f fraud_flag=%d",
                    transaction_id,
                    score,
                    fraud_flag,
                )

            except Exception:
                logger.exception("Failed to process Kafka message")

    finally:
        consumer.close()
        producer.flush()
        health_server.shutdown()
        logger.info("Scoring service stopped")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, stop_service)
    signal.signal(signal.SIGINT, stop_service)
    main()
