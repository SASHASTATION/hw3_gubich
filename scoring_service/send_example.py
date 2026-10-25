"""Generate one Kafka transaction to verify that the live scorer uses latest.txt."""
import json
import os
import uuid
from datetime import datetime, timezone

from confluent_kafka import Producer

transaction_id = f"hw3_{uuid.uuid4().hex[:12]}"
now = datetime.now(timezone.utc).isoformat(sep=" ")
message = {
    "transaction_id": transaction_id,
    "data": {
        "transaction_time": now, "amt": 175.0, "city_pop": 100000,
        "lat": 51.5, "long": 0.0, "merch_lat": 51.7, "merch_long": 0.3,
        "dob": "1990-01-01",
    },
}
producer = Producer({
    "bootstrap.servers": os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
})
producer.produce(
    os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions"),
    key=transaction_id.encode(), value=json.dumps(message).encode(),
)
producer.flush(10)
print("SENT_TRANSACTION_ID=" + transaction_id)
