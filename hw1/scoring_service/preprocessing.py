import math
from datetime import datetime

import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "amount_log",
    "population_log",
    "distance_log",
    "hour",
    "day_of_week",
    "is_night",
    "is_weekend",
    "age_years",
]


def _first(data: dict, *names, default=None):
    for name in names:
        value = data.get(name)
        if value is not None and not pd.isna(value):
            return value
    return default


def _to_float(value, default=0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_datetime(data: dict) -> pd.Timestamp:
    raw = _first(data, "transaction_time", "trans_date_trans_time")
    if raw is not None:
        parsed = pd.to_datetime(raw, errors="coerce")
        if not pd.isna(parsed):
            return parsed

    unix_time = _first(data, "unix_time")
    if unix_time is not None:
        parsed = pd.to_datetime(unix_time, unit="s", errors="coerce")
        if not pd.isna(parsed):
            return parsed

    return pd.Timestamp("1970-01-01 12:00:00")


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    values = [_to_float(x, default=np.nan) for x in (lat1, lon1, lat2, lon2)]
    if any(np.isnan(x) for x in values):
        return 0.0

    lat1, lon1, lat2, lon2 = map(math.radians, values)
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 6371.0088 * 2 * math.asin(math.sqrt(a))


def _age_years(data: dict, transaction_dt: pd.Timestamp) -> float:
    raw_dob = _first(data, "dob", "birth_date")
    if raw_dob is None:
        return 0.0

    dob = pd.to_datetime(raw_dob, errors="coerce")
    if pd.isna(dob):
        return 0.0

    age = (transaction_dt - dob).days / 365.25
    return float(max(0.0, min(age, 100.0)))


def preprocess_transaction(data: dict) -> dict:
    """Convert one raw Kafka transaction into model features."""
    transaction_dt = _parse_datetime(data)

    amount = max(0.0, _to_float(_first(data, "amount", "amt"), 0.0))
    population = max(
        0.0,
        _to_float(_first(data, "population_city", "city_pop"), 0.0),
    )

    distance = _haversine_km(
        _first(data, "lat"),
        _first(data, "lon", "long"),
        _first(data, "merchant_lat", "merch_lat"),
        _first(data, "merchant_lon", "merch_long"),
    )

    hour = int(transaction_dt.hour)
    day_of_week = int(transaction_dt.dayofweek)

    return {
        "amount_log": math.log1p(amount),
        "population_log": math.log1p(population),
        "distance_log": math.log1p(max(distance, 0.0)),
        "hour": float(hour),
        "day_of_week": float(day_of_week),
        "is_night": float(hour < 6 or hour >= 23),
        "is_weekend": float(day_of_week >= 5),
        "age_years": _age_years(data, transaction_dt),
    }


def to_feature_vector(features: dict) -> np.ndarray:
    return np.array([features[name] for name in FEATURE_NAMES], dtype=float)
