# TESTPROJ

Monorepo of three use cases: prompt similarity (usecase_1), call outcome prediction (usecase_2), and LLM response quality evaluation (usecase_3). Each has a backend API and a Streamlit frontend.

## Setup (from repo root)

```bash
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```



## usecase_1 — Prompt Similarity Service


**Backend (from root):**

```bash
cd usecase_1
uvicorn main:app --host 0.0.0.0 --port 8000
```

- API: http://localhost:8000  
- Docs: http://localhost:8000/docs  

**Frontend (from root):**

```bash
cd usecase_1
streamlit run streamlit_app.py
```

- UI: http://localhost:8501  

### Design Decisions and Tradeoffs

**Embedding Model Choice:**
- Selected `all-MiniLM-L6-v2` (384-dim) for balance of speed, quality, and memory. Tradeoff: larger models (e.g., `all-mpnet-base-v2`) would improve accuracy but increase latency and storage.

**Storage Architecture:**
- SQLite for both prompts and embeddings (separate DBs) for simplicity and portability. Tradeoff: PostgreSQL/vector DBs would scale better but add infrastructure complexity. BLOB storage for embeddings (int32 dim + float32 array) is space-efficient but requires deserialization overhead.

**Variable Normalization:**
- Normalize `{{variable}}` → `[VARIABLE_NAME]` before embedding to ensure template similarity regardless of variable names. Tradeoff: loses semantic meaning of variable names but ensures stable embeddings for template matching.

**Metadata-Aware Similarity Tiers:**
- Three-tier system (Tier1: same layer+category, Tier2: same layer, Tier3: cross-layer) with different thresholds (0.92, 0.90, 0.88). Tradeoff: more nuanced than single threshold but requires domain knowledge to tune thresholds.

**Caching Strategy:**
- In-memory TTL cache (5 minutes) for embeddings and similarity results. Tradeoff: fast but not distributed; Redis would enable multi-instance scaling but adds dependency.

**Layered Architecture:**
- Separated data, embedding, similarity, and API layers for testability. Tradeoff: more files but clearer separation of concerns.

### What Would Be Improved With More Time

1. **Vector Database Integration:** Migrate to Pinecone/Weaviate/Qdrant for scalable similarity search with approximate nearest neighbor (ANN) algorithms.
2. **Incremental Embedding Updates:** Support partial re-embedding when prompts change without full regeneration.
3. **Rate Limiting & Authentication:** Add proper rate limiting and API key authentication for production use.
4. **Model Versioning:** Track embedding model versions and support A/B testing of different models.
5. **Batch Similarity API:** Add endpoint for bulk similarity computation to reduce round-trips.
6. **Advanced Clustering:** Implement DBSCAN or HDBSCAN for automatic duplicate cluster discovery.
7. **Monitoring & Observability:** Add Prometheus metrics, logging, and alerting for production monitoring.

### Assumptions Made

1. **Prompt Format:** Assumes prompts use `{{variable}}` syntax; other templating formats (e.g., `{variable}`, `$variable`) not supported.
2. **Embedding Stability:** Assumes embedding model remains constant; changing models requires full re-embedding.
3. **Single Language:** Designed for English prompts; multilingual support would require different models.
4. **Local Deployment:** Optimized for single-instance deployment; distributed caching/load balancing not considered.
5. **Prompt Size:** Assumes prompts fit in memory; very large prompts (>10K tokens) may need chunking.
6. **Metadata Schema:** Assumes fixed metadata fields (layer, category, tier); schema changes require migration.

---

## usecase_2 — Call Outcome Prediction


**Backend (from root):**

```bash
cd usecase_2
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- API: http://localhost:8000  
- Endpoints: `/api/pipeline/run`, `/api/predict`, `/api/models`, etc.

**Frontend (from root):**

```bash
cd usecase_2
streamlit run frontend/app.py
```

- UI: http://localhost:8501 (expects API on port 8000)

### Design Decisions and Tradeoffs

**Ensemble Approach:**
- Combined XGBoost (tabular features) + LSTM (sequence modeling) for complementary strengths. Tradeoff: higher complexity and training time but better accuracy than single models. Simple weighted average ensemble; stacking/boosting would improve performance but add complexity.

**Feature Engineering:**
- 28 hand-crafted features from call events (timing, speech dynamics, engagement, progress, context). Tradeoff: domain expertise required but interpretable; deep learning on raw events would be more automated but less explainable.

**Sequence Modeling:**
- LSTM with fixed 50-timestep sequences, 8 event types, 2 continuous features (duration, words). Tradeoff: fixed length simplifies training but truncates long calls; variable-length sequences would be more accurate but require padding/masking complexity.

**Model Registry:**
- JSON-based registry with versioning (max 3 versions per type). Tradeoff: simple file-based approach; proper model registry (MLflow, Weights & Biases) would add features but require infrastructure.

**Training Pipeline:**
- Sequential pipeline (generate → validate → features → train XGB → train LSTM → ensemble). Tradeoff: easy to debug but not parallelized; parallel execution would speed up but complicate dependency management.

**Missing Value Handling:**
- Sentinel value (999.0) for "never occurred" timing features; 0 for other missing values. Tradeoff: simple but may confuse models; learned embeddings or imputation would be more sophisticated.

**SHAP Analysis:**
- Post-training SHAP for XGBoost feature importance. Tradeoff: adds dependency and compute time but provides interpretability.

### What Would Be Improved With More Time

1. **Real-time Prediction:** Add streaming prediction endpoint that updates as events arrive (currently requires full event sequence).
2. **Hyperparameter Tuning:** Implement automated hyperparameter optimization (Optuna, Ray Tune) for both XGBoost and LSTM.
3. **Cross-Validation:** Add proper k-fold cross-validation instead of single train/test split for more robust metrics.
4. **Model Monitoring:** Implement production monitoring with drift detection, prediction distribution tracking, and automated retraining triggers.
5. **Feature Store:** Build feature store for online/offline feature consistency and versioning.
6. **A/B Testing Framework:** Add infrastructure for deploying multiple model versions and comparing performance.
7. **Explainability Dashboard:** Enhance SHAP visualizations with interactive Streamlit dashboard showing prediction explanations.
8. **Data Augmentation:** Implement synthetic call generation for rare outcome classes to improve class balance.
9. **Transformer Models:** Experiment with transformer-based sequence models (BERT, GPT) for potentially better sequence understanding.
10. **Production Deployment:** Add Docker containers, Kubernetes manifests, and CI/CD pipelines.

### Assumptions Made

1. **Event Schema:** Assumes fixed event types (call_start, agent_speech, user_speech, silence, tool_call, call_end); new event types require code changes.
2. **Outcome Classes:** Fixed 4-class classification (completed, abandoned, transferred, error); adding classes requires retraining.
3. **Call Duration:** Assumes calls are <50 events or truncates; very long calls may lose important late-stage signals.
4. **Temporal Ordering:** Assumes events are chronologically ordered; out-of-order events would break feature computation.
5. **Metadata Availability:** Assumes metadata (agent_id, org_id, call_purpose) is always present; missing values default to empty strings.
6. **Training Data:** Assumes sufficient labeled data (500+ calls); model performance degrades with less data.
7. **Stationarity:** Assumes call patterns don't drift over time; production monitoring needed to detect concept drift.

---

## usecase_3 — LLM Response Quality Evaluator


**Backend (from root):**

```bash
cd usecase_3
export OPENAI_API_KEY=sk-...   # required for evaluate and improve
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- API: http://localhost:8000  
- Docs: http://localhost:8000/api/docs  
- Health: http://localhost:8000/health  

**Frontend (from root):**

```bash
cd usecase_3
streamlit run frontend/app.py
```

- UI: http://localhost:8501 (default API URL: http://localhost:8000)

**Endpoints:** `POST /api/evaluate`, `POST /api/evaluate/batch`, `POST /api/compare`, `POST /api/improve`

### Design Decisions and Tradeoffs

**LLM-as-Judge Architecture:**
- Uses OpenAI GPT-4o-mini as evaluator for flexibility and quality. Tradeoff: higher cost and latency vs. rule-based systems, but more nuanced evaluation. Temperature=0 for consistency but may miss creative edge cases.

**Multi-Dimensional Evaluation:**
- Six dimensions (task_completion, empathy, conciseness, naturalness, safety, clarity) with context-weighted aggregation. Tradeoff: comprehensive but requires careful weight tuning; simpler single-score would be faster but less actionable.

**Caching Strategy:**
- Hash-based caching (content hash) with TTL for identical requests. Tradeoff: reduces API costs but may serve stale results if evaluation criteria change; cache invalidation not implemented.

**Timeout & Fallback:**
- 5-second timeout with cached result fallback. Tradeoff: improves reliability but may return outdated evaluations; proper retry logic would be better.

**A/B Comparison Logic:**
- Dimension-wise winner detection (threshold=1) + weighted overall winner (threshold=0.5) + confidence scoring. Tradeoff: nuanced but complex; simpler majority-vote would be faster but less reliable.

**Batch Processing:**
- Sequential batch evaluation with cost estimation. Tradeoff: simple but slow; parallel evaluation would speed up but increase API rate limit risk.

**Drift Detection:**
- Weekly human sampling + Spearman correlation monitoring. Tradeoff: catches degradation but requires human annotation infrastructure; automated drift detection would be cheaper but less reliable.

**Schema Validation:**
- Pydantic schemas with strict validation. Tradeoff: catches errors early but may reject valid edge cases; looser validation would be more permissive but riskier.

### What Would Be Improved With More Time

1. **Evaluation Model Options:** Support multiple judge models (GPT-4, Claude, local models) with A/B testing and cost/quality tradeoff selection.
2. **Fine-Tuned Judge:** Train a smaller, specialized model on human-annotated data to reduce API costs and latency.
3. **Structured Output:** Use OpenAI's structured outputs (JSON mode) for more reliable parsing instead of regex-based extraction.
4. **Parallel Batch Processing:** Implement concurrent batch evaluation with rate limit handling and retry logic.
5. **Evaluation History:** Add database persistence for evaluation history, trend analysis, and audit trails.
6. **Custom Dimension Weights:** Allow per-request or per-context-type weight overrides instead of fixed weights.
7. **Confidence Calibration:** Implement Platt scaling or isotonic regression to calibrate confidence scores against human judgments.
8. **Multi-Model Comparison:** Extend A/B comparison to support comparing 3+ variants simultaneously.
9. **Improvement Iteration:** Add iterative improvement endpoint that refines responses over multiple rounds.
10. **Production Monitoring:** Add comprehensive observability (metrics, traces, logs) with alerting for degradation.
11. **Cost Optimization:** Implement request batching, model routing (cheaper models for simple cases), and usage analytics.
12. **Human-in-the-Loop:** Add workflow for human reviewers to provide feedback that improves the judge over time.

### Assumptions Made

1. **OpenAI API Availability:** Assumes OpenAI API is accessible and reliable; no fallback to alternative providers.
2. **Response Format:** Assumes LLM returns valid JSON; malformed responses require error handling (currently raises exceptions).
3. **Context Types:** Fixed set of context types (screening, appointment, follow_up, etc.); new types require schema updates.
4. **Dimension Stability:** Assumes evaluation dimensions remain constant; adding/removing dimensions requires prompt and schema changes.
5. **Human Annotation:** Drift detection assumes weekly human annotation is feasible; may not scale without annotation infrastructure.
6. **Conversation History:** Assumes conversation history is provided in correct format; malformed history may affect evaluation quality.
7. **Single Language:** Designed for English; multilingual evaluation would require language-specific prompts and validation.
8. **Response Length:** Assumes responses fit in model context window; very long responses may need truncation or chunking.
9. **Evaluation Latency:** 5-second timeout assumes typical API response time; slower networks may need longer timeouts.
10. **Cost Model:** Assumes API costs are acceptable; high-volume usage may require cost controls or alternative architectures.
