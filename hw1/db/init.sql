CREATE TABLE IF NOT EXISTS transaction_scores (
    transaction_id TEXT PRIMARY KEY,
    score DOUBLE PRECISION NOT NULL CHECK (score >= 0 AND score <= 1),
    fraud_flag SMALLINT NOT NULL CHECK (fraud_flag IN (0, 1)),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_transaction_scores_created_at
    ON transaction_scores (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_transaction_scores_fraud
    ON transaction_scores (fraud_flag, created_at DESC);
