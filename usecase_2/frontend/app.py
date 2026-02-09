"""
Streamlit frontend for Call Outcome Prediction (usecase_2).
Requires API running: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""
import json
import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Call Outcome Prediction", page_icon="📞", layout="wide")
st.title("Call Outcome Prediction")
st.caption("Pipeline, Predict, Models, Monitoring")

sidebar = st.sidebar
page = sidebar.radio(
    "Section",
    ["Pipeline", "Predict", "Models", "Monitoring"],
    label_visibility="collapsed",
)
sidebar.markdown("---")
sidebar.markdown("**API:** " + API_BASE)
if sidebar.button("Check API"):
    try:
        r = requests.get(f"{API_BASE}/api/models", timeout=5)
        sidebar.success("API OK" if r.ok else f"API {r.status_code}")
    except Exception as e:
        sidebar.error(str(e))


def call_api(method: str, path: str, **kwargs) -> tuple[bool, dict | str]:
    try:
        url = f"{API_BASE}{path}"
        if method == "GET":
            r = requests.get(url, timeout=300, **kwargs)
        else:
            r = requests.post(url, timeout=300, **kwargs)
        if r.status_code >= 400:
            return False, r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        return True, r.json() if r.content else {}
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to API. Start it with: cd usecase_2 && uvicorn api.main:app --port 8000"
    except Exception as e:
        return False, str(e)


# ----- Pipeline -----
if page == "Pipeline":
    st.header("Full pipeline")
    st.markdown("Run all steps: **Generate calls** → Validate → Feature engineering → Feature validation → **Train XGBoost** → SHAP → Partial sequences → **Train LSTM** → **Ensemble**.")
    if st.button("Run pipeline", type="primary"):
        with st.spinner("Running pipeline (this may take several minutes)..."):
            ok, out = call_api("POST", "/api/pipeline/run")
        if ok:
            st.success(out.get("message", "Done"))
            steps = out.get("steps", [])
            for s in steps:
                icon = "✅" if s.get("status") == "ok" else "❌"
                with st.expander(f"{icon} {s.get('name', '')}"):
                    st.code(s.get("message", ""))
        else:
            st.error(out if isinstance(out, str) else out.get("detail", str(out)))


# ----- Predict -----
if page == "Predict":
    st.header("Predict outcome")
    col1, col2 = st.columns(2)
    with col1:
        call_id = st.text_input("Call ID", value="call_demo_001")
        st.subheader("Events so far")
        events_json = st.text_area(
            "Events (JSON array)",
            value=json.dumps([
                {"ts": 0, "type": "call_start", "duration_ms": 0, "words": 0},
                {"ts": 5, "type": "user_speech", "duration_ms": 2000, "words": 30},
                {"ts": 8, "type": "agent_speech", "duration_ms": 3500, "words": 45},
            ], indent=2),
            height=200,
        )
    with col2:
        st.subheader("Metadata")
        agent_id = st.text_input("Agent ID", value="agent_a1")
        org_id = st.text_input("Org ID", value="org_1")
        call_purpose = st.selectbox("Call purpose", ["billing", "sdoh_screening", "support", "appointment_scheduling"])
        time_of_day = st.selectbox("Time of day", ["morning", "afternoon", "evening", "night"])
        day_of_week = st.selectbox("Day of week", ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])

    if st.button("Predict"):
        try:
            events = json.loads(events_json)
        except json.JSONDecodeError as e:
            st.error(f"Invalid events JSON: {e}")
        else:
            payload = {
                "call_id": call_id,
                "events_so_far": events,
                "metadata": {
                    "agent_id": agent_id,
                    "org_id": org_id,
                    "call_purpose": call_purpose,
                    "time_of_day": time_of_day,
                    "day_of_week": day_of_week,
                },
            }
            ok, out = call_api("POST", "/api/predict", json=payload)
            if ok:
                st.success(f"**Predicted outcome:** {out.get('predicted_outcome', '')} (confidence: {out.get('confidence', 0):.2f})")
                st.json(out)
            else:
                st.error(out if isinstance(out, str) else out.get("detail", str(out)))


# ----- Models -----
if page == "Models":
    st.header("Models")
    ok, out = call_api("GET", "/api/models")
    if not ok:
        st.error(out if isinstance(out, str) else str(out))
    else:
        models = out.get("models", [])
        if not models:
            st.info("No models registered. Run the pipeline or train via API.")
        else:
            st.dataframe(
                [{"model_id": m["model_id"], "type": m["type"], "accuracy": m.get("accuracy", 0), "status": m.get("status", "")} for m in models],
                use_container_width=True,
                hide_index=True,
            )
            selected = st.selectbox("Feature importance", [m["model_id"] for m in models], key="model_sel")
            if selected and st.button("Load importance"):
                ok2, imp = call_api("GET", f"/api/model/{selected}/importance")
                if ok2 and imp.get("global_importance"):
                    st.subheader(f"Global importance ({imp.get('importance_type', '')})")
                    st.dataframe(
                        [{"rank": x["rank"], "feature": x["feature"], "importance": x["importance"]} for x in imp["global_importance"][:15]],
                        use_container_width=True,
                        hide_index=True,
                    )
                    if imp.get("class_specific_importance"):
                        for cls, items in imp["class_specific_importance"].items():
                            with st.expander(f"Class: {cls}"):
                                st.dataframe(
                                    [{"feature": x["feature"], "contribution": x.get("mean_shap", x.get("contribution", 0))} for x in items[:10]],
                                    use_container_width=True,
                                    hide_index=True,
                                )
                elif ok2:
                    st.info("No importance data for this model.")
                else:
                    st.error(imp)


# ----- Monitoring -----
if page == "Monitoring":
    st.header("Daily monitoring report")
    report_date = st.text_input("Report date (YYYY-MM-DD, optional)", value="", placeholder="Leave empty for latest")
    if st.button("Generate report"):
        path = "/api/monitoring/report" + (f"?date={report_date}" if report_date.strip() else "")
        ok, out = call_api("GET", path)
        if ok:
            st.text_area("Report", value=out.get("report_text", ""), height=400)
            metrics = out.get("metrics", {})
            if metrics:
                with st.expander("Metrics (JSON)"):
                    st.json(metrics)
        else:
            st.error(out if isinstance(out, str) else str(out))
