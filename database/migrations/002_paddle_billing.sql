-- ============================================================
-- Migration 002 — Paddle billing columns
-- Apply after migration 001.
--
--   psql -U macropulse -d macropulse -f database/migrations/002_paddle_billing.sql
-- ============================================================

-- Rename stripe → paddle (safe: column may not exist yet)
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS paddle_customer_id      TEXT,
    ADD COLUMN IF NOT EXISTS paddle_subscription_id  TEXT;

-- Drop stripe column if it was created by migration 001
ALTER TABLE users
    DROP COLUMN IF EXISTS stripe_customer_id;
