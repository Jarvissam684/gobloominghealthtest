# Drift Detection and Retuning

Background monitoring for the LLM judge: weekly human samples, Spearman correlation (SCC), alerting, root cause analysis, and retuning validation.

## Contents

- **monitoring_protocol.md** — Full protocol: data collection, drift measurement, alerting rules, root cause analysis, automatic retuning, validation gate, deployment approval, confidence-based fallback. Includes decision trees and references to code.
- **drift_detection.py** — Implementations:
  - Stratified sampling (by `context_type`, score band)
  - Drift measurement using `calibration.validation` (Spearman, `drift_check`)
  - Alerting (SCC &lt; 0.75 alert, &lt; 0.65 hold, trend decline)
  - Root cause analysis (worst dimensions, systematic bias, by context_type)
  - Divergent case identification and retuning proposals
  - Retuning validation gate and deployment checks
  - Confidence-based escalation (should_escalate_to_human)
  - Weekly job stub: `run_weekly_drift_job`

## Running

From project root (`usecase_3`):

```bash
python -m monitoring.drift_detection
```

Uses dummy data; in production, replace with evaluations from your DB and collected human ratings.

## Dependencies

- `calibration.validation`: Spearman, consensus, `drift_check` (see `calibration/validation.py`).
- Optional: `scipy.stats.spearmanr` for Spearman; fallback implemented in calibration.

## Thresholds (protocol)

| Constant | Value | Meaning |
|----------|--------|--------|
| SCC_THRESHOLD_ALERT | 0.75 | Below → alert, prepare retuning |
| SCC_THRESHOLD_HOLD | 0.65 | Below → halt evals, manual review |
| SCC_THRESHOLD_DEPLOY | 0.80 | Required for deployment approval |
| SCC_THRESHOLD_VALIDATE | 0.78 | Retuning pass gate |
| CONFIDENCE_ESCALATION_THRESHOLD | 0.6 | Route to human when confidence &lt; this (fallback) |
