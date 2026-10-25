"""Data Quality checks and HTML report writer (no external reporting framework)."""
from __future__ import annotations

import html
from collections import Counter
from datetime import datetime, timezone

from pipeline.common import REPORT_DIR, airflow_run_key, atomic_write_text, connect


def generate_report() -> None:
    """Read current labeled showcase; write timestamped HTML in sibling reports/.

    Reports use stable Airflow run hash (retry overwrites same file); no showcase
    rows are modified by the DQ DAG. A failed check stays visible in HTML.
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM labeled_transactions")
            total = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM labeled_transactions WHERE is_fraud IS NULL OR is_fraud NOT IN (0,1)")
            bad_labels = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM labeled_transactions WHERE transaction_time IS NULL")
            missing_time = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM labeled_transactions WHERE payload IS NULL")
            missing_payload = cur.fetchone()[0]
            cur.execute("""SELECT COUNT(*) FROM labeled_transactions
                            WHERE NOT (payload ? 'amt') OR NOT (payload ? 'city_pop')""")
            missing_features = cur.fetchone()[0]
            cur.execute("""SELECT COUNT(*) FROM labeled_transactions
                           WHERE (payload->>'amt') IS NULL
                              OR (payload->>'amt') !~ '^[0-9]+(\\.[0-9]+)?$'""")
            invalid_amount = cur.fetchone()[0]
            cur.execute("""SELECT COUNT(*) FROM
                            (SELECT sample_key,batch_month,COUNT(*)
                             FROM labeled_transactions GROUP BY 1,2 HAVING COUNT(*)>1) x""")
            duplicate_keys = cur.fetchone()[0]
            cur.execute("""SELECT COUNT(*) FROM labeled_transactions
                           WHERE batch_month<>date_trunc('month', transaction_time)::date""")
            month_mismatch = cur.fetchone()[0]
            cur.execute("""SELECT COUNT(*) FROM labeled_transactions
                           WHERE (payload->>'lat') IS NULL OR (payload->>'long') IS NULL
                              OR (payload->>'merch_lat') IS NULL OR (payload->>'merch_long') IS NULL""")
            missing_coordinates = cur.fetchone()[0]
            cur.execute("SELECT MIN(batch_month), MAX(batch_month) FROM labeled_transactions")
            range_first, range_last = cur.fetchone()
    checks = [
        ("DQ01: витрина не пустая", total > 0, f"rows={total}"),
        ("DQ02: fraud label ∈ {0, 1}", bad_labels == 0, f"violations={bad_labels}"),
        ("DQ03: transaction_time не NULL", missing_time == 0, f"violations={missing_time}"),
        ("DQ04: payload присутствует", missing_payload == 0, f"violations={missing_payload}"),
        ("DQ05: обязательные признаки amt/city_pop", missing_features == 0, f"violations={missing_features}"),
        ("DQ06: сумма транзакции валидна", invalid_amount == 0, f"violations={invalid_amount}"),
        ("DQ07: (sample_key, batch_month) уникален", duplicate_keys == 0, f"violations={duplicate_keys}"),
        ("DQ08: transaction_time в указанном месяце", month_mismatch == 0, f"violations={month_mismatch}"),
        ("DQ09: координаты присутствуют", missing_coordinates == 0, f"violations={missing_coordinates}"),
    ]
    passed = sum(ok for _, ok, _ in checks)
    body = "\n".join(
        f"<tr><td>{html.escape(name)}</td><td class='{('ok' if ok else 'bad')}'>{'PASS' if ok else 'FAIL'}</td><td>{html.escape(detail)}</td></tr>"
        for name, ok, detail in checks
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report = f"""<!doctype html><html lang='ru'><head><meta charset='utf-8'>
<title>Fraud dataset quality</title>
<style>body{{font:16px sans-serif;max-width:980px;margin:30px auto;padding:0 15px}}
table{{width:100%;border-collapse:collapse}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}
.ok{{color:#157b36;font-weight:bold}}.bad{{color:#b72b2b;font-weight:bold}}</style></head>
<body><h1>Fraud labeled showcase: Data Quality</h1>
<p>Run: {html.escape(airflow_run_key())}</p><p>Created: {timestamp}</p>
<p>Months: {range_first} — {range_last}; total rows: {total}</p>
<h2>Passed: {passed}/{len(checks)}</h2>
<table><thead><tr><th>Rule</th><th>Status</th><th>Details</th></tr></thead>
<tbody>{body}</tbody></table></body></html>"""
    # Unique, stable run ID; not tied to wall clock -> retries do not create duplicates.
    import hashlib
    run_hash = hashlib.sha256(airflow_run_key().encode()).hexdigest()[:14]
    target = REPORT_DIR / f"dq_{run_hash}.html"
    atomic_write_text(target, report)
    atomic_write_text(REPORT_DIR / "latest.html", report)
    print(f"DQ report: {target}, passed {passed}/{len(checks)}")
    if passed != len(checks):
        raise ValueError(f"Data quality violations: {len(checks) - passed} failed checks; HTML saved")
