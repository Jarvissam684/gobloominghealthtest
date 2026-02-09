#!/usr/bin/env python3
"""
Single pipeline: generate calls → validate → feature engineering → feature validation →
train XGBoost → SHAP analysis → partial sequences → train LSTM → ensemble.
Runs all steps in order; stops on first failure.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

PIPELINE_STEPS = [
    ("generate_calls", "generate_calls.py", "Generate 500 synthetic calls"),
    ("validate_calls", "validate_calls.py", "Validate schema, sequence, correlation"),
    ("feature_engineering", "feature_engineering.py", "Compute 28 features → call_features.json"),
    ("feature_validation_report", "feature_validation_report.py", "Feature validation report"),
    ("train_xgb", "train_xgb.py", "Train XGBoost classifier"),
    ("shap_analysis", "shap_analysis.py", "SHAP importance analysis"),
    ("partial_sequences", "partial_sequences.py", "Build 2500 partial sequences"),
    ("train_lstm", "train_lstm.py", "Train LSTM classifier"),
    ("ensemble_pipeline", "ensemble_pipeline.py", "Ensemble pipeline & evaluation"),
]


def run_step(name: str, script: str, description: str, timeout: int = 300) -> dict:
    """Run a single pipeline step via subprocess. Returns {name, status, message, returncode}."""
    path = BASE_DIR / script
    if not path.exists():
        return {"name": name, "status": "fail", "message": f"Script not found: {script}", "returncode": -1}
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        if result.returncode != 0:
            msg = err or out or f"Exit code {result.returncode}"
            return {"name": name, "status": "fail", "message": msg[:500], "returncode": result.returncode}
        return {"name": name, "status": "ok", "message": out[:300] if out else "OK", "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        return {"name": name, "status": "fail", "message": f"Timeout after {timeout}s", "returncode": -1}
    except Exception as e:
        return {"name": name, "status": "fail", "message": str(e)[:500], "returncode": -1}


def run_pipeline(timeout_per_step: int = 300) -> dict:
    """
    Run full pipeline. Returns:
    {
        "success": bool,
        "steps": [{"name", "status", "message", "returncode"}, ...],
        "message": "All steps OK" or "Failed at step X"
    }
    """
    steps_out = []
    for name, script, _desc in PIPELINE_STEPS:
        step = run_step(name, script, _desc, timeout=timeout_per_step)
        steps_out.append(step)
        if step["status"] != "ok":
            return {
                "success": False,
                "steps": steps_out,
                "message": f"Pipeline failed at step: {name}",
            }
    return {
        "success": True,
        "steps": steps_out,
        "message": "All steps completed successfully",
    }


def main():
    result = run_pipeline()
    for s in result["steps"]:
        symbol = "✓" if s["status"] == "ok" else "✗"
        print(f"  {symbol} {s['name']}: {s['message'][:80]}...")
    print(result["message"])
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
