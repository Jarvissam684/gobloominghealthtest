-- =============================================================================
-- Dashboard queries for LLM Response Quality Evaluator
-- Assumes PostgreSQL-style syntax. Adapt table names and types to your logging DB.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Schema (reference; create in your logging DB)
-- -----------------------------------------------------------------------------
/*
CREATE TABLE request_log (
  id           BIGSERIAL PRIMARY KEY,
  ts           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  endpoint     TEXT NOT NULL,        -- '/api/evaluate', '/api/evaluate/batch', '/api/compare', '/api/improve'
  method       TEXT NOT NULL,
  status_code  INT NOT NULL,
  latency_ms   NUMERIC(12,2),
  error_message TEXT,
  meta         JSONB                 -- eval_id, agent_id, prompt_version, context_type, source, batch_id, cost_usd
);

CREATE TABLE evaluations (
  eval_id        TEXT PRIMARY KEY,
  request_id     BIGINT REFERENCES request_log(id),
  overall_score  NUMERIC(4,2) NOT NULL,
  dimensions     JSONB NOT NULL,     -- { "task_completion": { "score": 8, "confidence": 0.9 }, ... }
  flags         JSONB DEFAULT '[]',  -- [ { "code": "safety_low", "message": "...", "severity": "high" } ]
  agent_id       TEXT,
  prompt_version TEXT,
  context_type   TEXT,
  source         TEXT DEFAULT 'live', -- 'live' | 'cache'
  cost_usd       NUMERIC(10,6) DEFAULT 0,
  created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE comparisons (
  id                   BIGSERIAL PRIMARY KEY,
  request_id            BIGINT REFERENCES request_log(id),
  eval_id_a             TEXT NOT NULL,
  eval_id_b             TEXT NOT NULL,
  winner                TEXT NOT NULL,  -- 'a' | 'b' | 'tie'
  confidence_in_winner  NUMERIC(4,3),
  context_type          TEXT,
  created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE drift_snapshots (
  week_start      DATE PRIMARY KEY,
  scc_overall     NUMERIC(5,3),
  scc_dimensions  JSONB,             -- { "task_completion": 0.82, "empathy": 0.78, ... }
  status          TEXT NOT NULL,      -- 'ok' | 'alert' | 'hold'
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_request_log_ts ON request_log(ts);
CREATE INDEX idx_request_log_endpoint ON request_log(endpoint);
CREATE INDEX idx_evaluations_created ON evaluations(created_at);
CREATE INDEX idx_evaluations_agent ON evaluations(agent_id);
CREATE INDEX idx_evaluations_prompt_version ON evaluations(prompt_version);
CREATE INDEX idx_evaluations_context_type ON evaluations(context_type);
CREATE INDEX idx_comparisons_created ON comparisons(created_at);
*/

-- =============================================================================
-- 1. SYSTEM HEALTH DASHBOARD
-- Refresh: 60s. Windows: 24h, 7d.
-- =============================================================================

-- 1.1 Request volume (last 24h, hourly buckets)
SELECT date_trunc('hour', ts) AS bucket,
       count(*) AS request_volume
FROM request_log
WHERE ts >= NOW() - INTERVAL '24 hours'
GROUP BY 1
ORDER BY 1;

-- 1.2 Success rate (last 24h)
SELECT count(*) FILTER (WHERE status_code = 200)::float / nullif(count(*), 0) * 100 AS success_rate_pct,
       count(*) AS total
FROM request_log
WHERE ts >= NOW() - INTERVAL '24 hours';

-- 1.3 Error rate (last 24h and last 15 min for alert)
SELECT count(*) FILTER (WHERE status_code IN (400, 408, 500))::float / nullif(count(*), 0) * 100 AS error_rate_pct
FROM request_log
WHERE ts >= NOW() - INTERVAL '24 hours';

-- For alert (last 15 min):
-- SELECT count(*) FILTER (WHERE status_code IN (400, 408, 500))::float / nullif(count(*), 0) * 100 AS error_rate_pct
-- FROM request_log WHERE ts >= NOW() - INTERVAL '15 minutes';

-- 1.4 Latency p50 / p99 (successful, last 24h)
SELECT percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_latency_ms,
       percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_latency_ms
FROM request_log
WHERE ts >= NOW() - INTERVAL '24 hours'
  AND status_code = 200
  AND latency_ms IS NOT NULL;

-- 1.5 Cache hit rate (evaluate requests only; last 1h for alert)
-- From request_log meta: source = 'cache' vs 'live'
SELECT count(*) FILTER (WHERE (meta->>'source') = 'cache')::float
       / nullif(count(*), 0) * 100 AS cache_hit_rate_pct
FROM request_log
WHERE ts >= NOW() - INTERVAL '24 hours'
  AND endpoint = '/api/evaluate'
  AND status_code = 200;

-- If evaluations table has source:
SELECT count(*) FILTER (WHERE e.source = 'cache')::float / nullif(count(*), 0) * 100 AS cache_hit_rate_pct
FROM request_log r
JOIN evaluations e ON e.request_id = r.id
WHERE r.ts >= NOW() - INTERVAL '24 hours'
  AND r.endpoint = '/api/evaluate';

-- 1.6 Cost anomaly: current 24h cost vs 7d rolling average
WITH cost_24h AS (
  SELECT coalesce(sum((meta->>'cost_usd')::numeric), 0) + coalesce(sum(e.cost_usd), 0) AS total
  FROM request_log r
  LEFT JOIN evaluations e ON e.request_id = r.id
  WHERE r.ts >= NOW() - INTERVAL '24 hours'
),
cost_7d AS (
  SELECT coalesce(sum((meta->>'cost_usd')::numeric), 0) + coalesce(sum(e.cost_usd), 0) AS total
  FROM request_log r
  LEFT JOIN evaluations e ON e.request_id = r.id
  WHERE r.ts >= NOW() - INTERVAL '7 days'
)
SELECT c24.total AS cost_24h,
       c7.total / 7 AS avg_daily_cost_7d,
       CASE WHEN (c7.total / 7) > 0 AND c24.total > 2 * (c7.total / 7) THEN true ELSE false END AS cost_anomaly
FROM cost_24h c24, cost_7d c7;

-- 1.7 Trend: Request volume by hour (24h)
SELECT date_trunc('hour', ts) AS bucket,
       count(*) AS volume
FROM request_log
WHERE ts >= NOW() - INTERVAL '24 hours'
GROUP BY 1
ORDER BY 1;

-- 1.8 Trend: Latency p50/p99 by hour (24h)
SELECT date_trunc('hour', ts) AS bucket,
       percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,
       percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99_ms
FROM request_log
WHERE ts >= NOW() - INTERVAL '24 hours'
  AND status_code = 200
  AND latency_ms IS NOT NULL
GROUP BY 1
ORDER BY 1;


-- =============================================================================
-- 2. EVALUATION QUALITY DASHBOARD
-- Refresh: 5m. Windows: 24h, 7d, 30d.
-- =============================================================================

-- 2.1 Per-dimension: mean score (last 7d)
SELECT (dimensions->'task_completion'->>'score')::numeric AS task_completion,
       (dimensions->'empathy'->>'score')::numeric AS empathy,
       (dimensions->'conciseness'->>'score')::numeric AS conciseness,
       (dimensions->'naturalness'->>'score')::numeric AS naturalness,
       (dimensions->'safety'->>'score')::numeric AS safety,
       (dimensions->'clarity'->>'score')::numeric AS clarity
FROM evaluations
WHERE created_at >= NOW() - INTERVAL '7 days';

-- Aggregated mean per dimension:
SELECT avg((dimensions->'task_completion'->>'score')::numeric) AS mean_task_completion,
       avg((dimensions->'empathy'->>'score')::numeric) AS mean_empathy,
       avg((dimensions->'conciseness'->>'score')::numeric) AS mean_conciseness,
       avg((dimensions->'naturalness'->>'score')::numeric) AS mean_naturalness,
       avg((dimensions->'safety'->>'score')::numeric) AS mean_safety,
       avg((dimensions->'clarity'->>'score')::numeric) AS mean_clarity
FROM evaluations
WHERE created_at >= NOW() - INTERVAL '7 days';

-- 2.2 Per-dimension: histogram (e.g. task_completion 1-2, 3-4, ..., 9-10)
SELECT width_bucket((dimensions->'task_completion'->>'score')::numeric, 1, 10, 5) AS bucket,
       count(*) AS cnt
FROM evaluations
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY 1
ORDER BY 1;

-- 2.3 Per-agent: volume, mean overall_score, flag rate
SELECT agent_id,
       count(*) AS volume,
       round(avg(overall_score)::numeric, 2) AS mean_overall_score,
       count(*) FILTER (WHERE jsonb_array_length(flags) > 0)::float / nullif(count(*), 0) * 100 AS flag_rate_pct
FROM evaluations
WHERE created_at >= NOW() - INTERVAL '7 days'
  AND agent_id IS NOT NULL
GROUP BY agent_id
ORDER BY volume DESC;

-- 2.4 Per-prompt-version: volume, mean overall
SELECT prompt_version,
       count(*) AS volume,
       round(avg(overall_score)::numeric, 2) AS mean_overall_score
FROM evaluations
WHERE created_at >= NOW() - INTERVAL '7 days'
  AND prompt_version IS NOT NULL
GROUP BY prompt_version
ORDER BY volume DESC;

-- 2.5 Drift: weekly SCC (from drift_snapshots)
SELECT week_start,
       scc_overall,
       scc_dimensions,
       status
FROM drift_snapshots
ORDER BY week_start DESC
LIMIT 12;


-- =============================================================================
-- 3. COST & EFFICIENCY DASHBOARD
-- Refresh: 5m. Windows: 24h, 7d.
-- =============================================================================

-- 3.1 Cost per evaluation (last 24h)
SELECT sum(cost_usd) / nullif(count(*), 0) AS cost_per_eval_usd,
       count(*) AS total_evals
FROM evaluations
WHERE created_at >= NOW() - INTERVAL '24 hours';

-- 3.2 Cache hit rate (evaluations, last 24h)
SELECT count(*) FILTER (WHERE source = 'cache')::float / nullif(count(*), 0) * 100 AS cache_hit_rate_pct
FROM evaluations
WHERE created_at >= NOW() - INTERVAL '24 hours';

-- 3.3 LLM calls per second (live-only requests; approximate from request_log)
SELECT count(*)::float / 60 AS llm_calls_per_min
FROM request_log
WHERE ts >= NOW() - INTERVAL '1 minute'
  AND endpoint IN ('/api/evaluate', '/api/compare', '/api/improve')
  AND (meta->>'source') IS DISTINCT FROM 'cache';

-- 3.4 Cost by operation (endpoint)
SELECT endpoint,
       coalesce(sum((meta->>'cost_usd')::numeric), 0) AS cost_usd
FROM request_log
WHERE ts >= NOW() - INTERVAL '7 days'
GROUP BY endpoint
ORDER BY cost_usd DESC;

-- 3.5 Cost by model (if stored in meta; else default)
SELECT coalesce(meta->>'model', '4o-mini') AS model,
       sum(coalesce((meta->>'cost_usd')::numeric, 0)) + sum(coalesce(e.cost_usd, 0)) AS cost_usd
FROM request_log r
LEFT JOIN evaluations e ON e.request_id = r.id
WHERE r.ts >= NOW() - INTERVAL '7 days'
GROUP BY coalesce(meta->>'model', '4o-mini')
ORDER BY cost_usd DESC;

-- 3.6 Trend: Cost per evaluation (daily, 7d)
SELECT date_trunc('day', created_at)::date AS day,
       sum(cost_usd) / nullif(count(*), 0) AS cost_per_eval_usd
FROM evaluations
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY 1
ORDER BY 1;


-- =============================================================================
-- 4. FLAGGED RESPONSES DASHBOARD
-- Refresh: 5m. Windows: 24h, 7d.
-- =============================================================================

-- 4.1 Flag volume and flag rate (last 24h)
SELECT count(*) FILTER (WHERE jsonb_array_length(flags) > 0) AS flagged_volume,
       count(*) AS total_evals,
       count(*) FILTER (WHERE jsonb_array_length(flags) > 0)::float / nullif(count(*), 0) * 100 AS flag_rate_pct
FROM evaluations
WHERE created_at >= NOW() - INTERVAL '24 hours';

-- 4.2 Flag types (count per code; expand flags array)
SELECT elem->>'code' AS flag_code,
       count(*) AS cnt
FROM evaluations,
     jsonb_array_elements(flags) AS elem
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY elem->>'code'
ORDER BY cnt DESC;

-- 4.3 Flag rate by context_type
SELECT context_type,
       count(*) AS total,
       count(*) FILTER (WHERE jsonb_array_length(flags) > 0)::float / nullif(count(*), 0) * 100 AS flag_rate_pct
FROM evaluations
WHERE created_at >= NOW() - INTERVAL '7 days'
  AND context_type IS NOT NULL
GROUP BY context_type
ORDER BY total DESC;

-- 4.4 Recent flagged examples (score < 4 or has safety flag)
SELECT eval_id,
       overall_score,
       context_type,
       flags,
       left((dimensions->'task_completion'->>'reasoning'), 100) AS snippet
FROM evaluations
WHERE created_at >= NOW() - INTERVAL '24 hours'
  AND (overall_score < 4 OR flags @> '[{"code": "safety_low"}]'::jsonb OR jsonb_array_length(flags) > 0)
ORDER BY created_at DESC
LIMIT 50;

-- 4.5 Flag rate trend (daily, 7d) — for "is flag rate increasing?"
SELECT date_trunc('day', created_at)::date AS day,
       count(*) FILTER (WHERE jsonb_array_length(flags) > 0)::float / nullif(count(*), 0) * 100 AS flag_rate_pct
FROM evaluations
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY 1
ORDER BY 1;


-- =============================================================================
-- 5. A/B TESTING DASHBOARD
-- Refresh: 5m. Experiment window or last 7d/30d.
-- =============================================================================

-- 5.1 Overall winner and confidence (last 7d)
SELECT winner,
       count(*) AS cnt,
       round(avg(confidence_in_winner)::numeric, 3) AS avg_confidence
FROM comparisons
WHERE created_at >= NOW() - INTERVAL '7 days'
GROUP BY winner;

-- 5.2 Sample size n
SELECT count(*) AS n_comparisons
FROM comparisons
WHERE created_at >= NOW() - INTERVAL '7 days';

-- 5.3 Bias check: a_wins % vs b_wins %
SELECT count(*) FILTER (WHERE winner = 'a')::float / nullif(count(*), 0) * 100 AS a_wins_pct,
       count(*) FILTER (WHERE winner = 'b')::float / nullif(count(*), 0) * 100 AS b_wins_pct,
       count(*) FILTER (WHERE winner = 'tie')::float / nullif(count(*), 0) * 100 AS tie_pct
FROM comparisons
WHERE created_at >= NOW() - INTERVAL '7 days';

-- 5.4 Per-dimension comparison (requires dimension-level winner stored; if not, use eval scores from evaluations)
-- If comparisons store dimension winners in JSONB:
-- SELECT (dimensions->'task_completion'->>'winner') AS tc_winner, ...
-- Otherwise join comparisons to evaluations and compute mean score_a vs score_b per dimension:
SELECT c.id,
       c.winner,
       c.confidence_in_winner,
       ea.overall_score AS score_a,
       eb.overall_score AS score_b,
       (ea.dimensions->'empathy'->>'score')::int AS empathy_a,
       (eb.dimensions->'empathy'->>'score')::int AS empathy_b
FROM comparisons c
JOIN evaluations ea ON ea.eval_id = c.eval_id_a
JOIN evaluations eb ON eb.eval_id = c.eval_id_b
WHERE c.created_at >= NOW() - INTERVAL '7 days';

-- Aggregated per-dimension winner (example for empathy):
SELECT CASE
         WHEN avg((ea.dimensions->'empathy'->>'score')::numeric) > avg((eb.dimensions->'empathy'->>'score')::numeric) THEN 'a'
         WHEN avg((eb.dimensions->'empathy'->>'score')::numeric) > avg((ea.dimensions->'empathy'->>'score')::numeric) THEN 'b'
         ELSE 'tie'
       END AS empathy_winner,
       round(avg((ea.dimensions->'empathy'->>'score')::numeric), 2) AS mean_empathy_a,
       round(avg((eb.dimensions->'empathy'->>'score')::numeric), 2) AS mean_empathy_b
FROM comparisons c
JOIN evaluations ea ON ea.eval_id = c.eval_id_a
JOIN evaluations eb ON eb.eval_id = c.eval_id_b
WHERE c.created_at >= NOW() - INTERVAL '7 days';
