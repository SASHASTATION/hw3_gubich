-- Applied on every `docker compose up`: ALTER works for existing HW1 volumes too.
ALTER TABLE transaction_scores
  ADD COLUMN IF NOT EXISTS model_version TEXT NOT NULL DEFAULT 'baseline_v1';

CREATE TABLE IF NOT EXISTS labeled_transactions (
    id BIGSERIAL PRIMARY KEY,
    sample_key TEXT NOT NULL,
    batch_month DATE NOT NULL,
    transaction_time TIMESTAMP NOT NULL,
    payload JSONB NOT NULL,
    is_fraud SMALLINT NOT NULL CHECK (is_fraud IN (0, 1)),
    UNIQUE (sample_key, batch_month)
);
CREATE INDEX IF NOT EXISTS idx_labeled_transactions_month
  ON labeled_transactions(batch_month, transaction_time);

CREATE TABLE IF NOT EXISTS supply_state (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  last_month DATE NOT NULL
);
CREATE TABLE IF NOT EXISTS supply_batches (
  run_key TEXT PRIMARY KEY,
  source_month DATE NOT NULL,
  target_month DATE NOT NULL,
  status TEXT NOT NULL DEFAULT 'claimed',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS training_state (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  last_model_month DATE,
  active_version TEXT
);
INSERT INTO training_state(id) VALUES (1) ON CONFLICT DO NOTHING;
CREATE TABLE IF NOT EXISTS training_runs (
  run_key TEXT PRIMARY KEY,
  target_month DATE NOT NULL,
  status TEXT NOT NULL DEFAULT 'claimed',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE TABLE IF NOT EXISTS model_history (
  version TEXT PRIMARY KEY,
  target_month DATE NOT NULL,
  candidate_ap DOUBLE PRECISION NOT NULL,
  incumbent_ap DOUBLE PRECISION,
  approved BOOLEAN NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
