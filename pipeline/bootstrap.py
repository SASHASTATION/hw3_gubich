"""Small reproducible labeled dataset for a *self-contained* homework demo.

Not intended to model real fraud. If replacing with Kaggle CSV, keep columns
(sample_key, batch_month, transaction_time, payload, is_fraud) in showcase.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np
from psycopg2.extras import Json, execute_values

from pipeline.common import add_months, connect, month_start

ROWS_PER_MONTH = 2000


def build_seed_rows(month: date, month_idx: int) -> list[tuple]:
    """Generate labeled raw rows with reproducible numeric fields for our own preprocessing."""
    rng = np.random.default_rng(20261021 + month_idx)
    start = datetime(month.year, month.month, 1)
    end = datetime(add_months(month, 1).year, add_months(month, 1).month, 1)
    days = (end - start).days
    output = []
    for i in range(ROWS_PER_MONTH):
        time = start + timedelta(
            days=int(rng.integers(0, days)),
            hours=int(rng.integers(0, 24)),
            minutes=int(rng.integers(0, 60)),
        )
        amount = round(float(rng.lognormal(4.0, 1.0)), 2)
        population = int(rng.integers(1000, 2_000_000))
        lat = float(rng.uniform(40, 55))
        lon = float(rng.uniform(20, 45))
        far = bool(rng.random() < 0.17)
        merchant_lat = lat + float(rng.normal(0, 2.0 if far else 0.05))
        merchant_lon = lon + float(rng.normal(0, 2.0 if far else 0.05))
        night = time.hour < 6 or time.hour >= 23
        # No label leakage: only input features determine the synthetic class probability.
        risk_logit = -3.3 + 0.30 * math.log1p(amount) + 0.80 * far + 0.55 * night
        probability = 1 / (1 + math.exp(-risk_logit))
        is_fraud = int(rng.random() < probability)
        birthday = date(1970 + int(rng.integers(0, 35)), 4, 13)
        payload = {
            "transaction_time": time.isoformat(sep=" "),
            "trans_date_trans_time": time.isoformat(sep=" "),
            "amt": amount,
            "city_pop": population,
            "lat": lat,
            "long": lon,
            "merch_lat": merchant_lat,
            "merch_long": merchant_lon,
            "dob": birthday.isoformat(),
        }
        output.append((f"seed_{month_idx}_{i}", month, time, Json(payload), is_fraud))
    return output


def main() -> None:
    """Read showcase count; seed 3 months once; repeated container starts never duplicate rows."""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM labeled_transactions")
            total = cur.fetchone()[0]
            if total:
                print(f"Labeled showcase already contains {total} rows; skip seeding")
                return
            newest_month = month_start(date.today())
            for month_idx, offset in enumerate((-2, -1, 0)):
                month = add_months(newest_month, offset)
                rows = build_seed_rows(month, month_idx)
                execute_values(
                    cur,
                    """INSERT INTO labeled_transactions
                         (sample_key, batch_month, transaction_time, payload, is_fraud)
                         VALUES %s ON CONFLICT (sample_key, batch_month) DO NOTHING""",
                    rows,
                    page_size=500,
                )
            cur.execute(
                "INSERT INTO supply_state(id, last_month) VALUES (1,%s) ON CONFLICT DO NOTHING",
                (newest_month,),
            )
            print(f"Seeded 3 months, {ROWS_PER_MONTH} labeled rows each, through {newest_month}")


if __name__ == "__main__":
    main()
