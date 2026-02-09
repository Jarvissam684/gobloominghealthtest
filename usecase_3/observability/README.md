# Observability Dashboards

Real-time visibility into the LLM Response Quality Evaluator: system health, evaluation quality, cost, flags, and A/B testing.

## Contents

| File | Description |
|------|-------------|
| **dashboard_design.md** | Dashboard wireframes (section layout), metric definitions, refresh intervals, alert thresholds. |
| **dashboard_queries.sql** | PostgreSQL-style SQL to populate dashboards from the logging DB (schema + queries for all 5 dashboards). |
| **README.md** | This file. |

## Dashboards Summary

| Dashboard | Purpose | Refresh | Key metrics |
|-----------|----------|---------|-------------|
| **System Health** | Request volume, success/error rate, latency, cache, cost anomaly | 60s | Volume, p50/p99, error rate >5% alert, latency >4s, cache <25% |
| **Evaluation Quality** | Per-dimension scores, per-agent/per-prompt, drift (SCC) | 5m | Mean score, histogram, flag rate, weekly SCC |
| **Cost & Efficiency** | Cost per eval, cache hit rate, LLM calls/s, by model/operation | 5m | $/eval, cache hit rate, cost by endpoint |
| **Flagged Responses** | Flag volume, types, by context_type, recent examples, trend | 5m | Flag rate, “is flag rate increasing?” |
| **A/B Testing** | Experiments (e.g. v2.1 vs v3.0), winner, per-dimension, confidence | 5m | Overall winner, confidence_in_winner, bias alert >65% |

## Logging Schema

Ensure your logging pipeline writes to (or can be queried as):

- **request_log** — Every API request: `ts`, `endpoint`, `status_code`, `latency_ms`, `meta` (eval_id, agent_id, prompt_version, context_type, source, cost_usd).
- **evaluations** — Per evaluation: `eval_id`, `overall_score`, `dimensions` (JSON), `flags` (JSON), `agent_id`, `prompt_version`, `context_type`, `source`, `cost_usd`, `created_at`.
- **comparisons** — Per compare: `eval_id_a`, `eval_id_b`, `winner`, `confidence_in_winner`, `context_type`, `created_at`.
- **drift_snapshots** — Weekly: `week_start`, `scc_overall`, `scc_dimensions` (JSON), `status`.

See the commented DDL at the top of `dashboard_queries.sql`.

## Alerts (Summary)

- **Error rate** > 5% (15 min) → notify on-call  
- **p99 latency** > 4 s (15 min) → notify  
- **Cache hit rate** < 25% (1 h) → info  
- **Cost** 24h > 2× 7d avg → warning  
- **Bias (A/B)** either side > 65% → alert  
