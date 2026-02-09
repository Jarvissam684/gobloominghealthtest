# Call Outcome Prediction (usecase_2)

Single pipeline, REST API, and Streamlit frontend for training and predicting call outcomes.

## Setup

```bash
cd usecase_2
pip install -r requirements.txt
```

## 1. Run full pipeline (CLI)

Runs in order: generate calls → validate → feature engineering → feature validation → train XGBoost → SHAP → partial sequences → train LSTM → ensemble.

```bash
python pipeline.py
```

## 2. Start API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `POST /api/pipeline/run` – run full pipeline
- `POST /api/model/train` – train model (xgboost | lstm | ensemble)
- `POST /api/predict` – predict from events + metadata
- `GET /api/models` – list models
- `GET /api/model/{model_id}/importance` – feature importance
- `GET /api/monitoring/report` – daily monitoring report

## 3. Start Streamlit frontend

With the API running on port 8000:

```bash
streamlit run frontend/app.py
```

Then open the URL shown (default http://localhost:8501). Use the sidebar: **Pipeline** (run pipeline), **Predict** (submit call events), **Models** (list and importance), **Monitoring** (daily report).

## One-shot: pipeline then API

```bash
python pipeline.py && uvicorn api.main:app --host 0.0.0.0 --port 8000
```

In another terminal: `streamlit run frontend/app.py`
