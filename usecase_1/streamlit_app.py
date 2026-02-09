"""
Prompt Similarity Service — Streamlit frontend.

Run: streamlit run streamlit_app.py
Requires the API server: uvicorn main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import httpx

# Default API URL (override in sidebar)
DEFAULT_API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Prompt Similarity Service",
    page_icon="📋",
    layout="wide",
)


def api_url() -> str:
    return st.session_state.get("api_url", DEFAULT_API_URL)


def get_status():
    try:
        r = httpx.get(f"{api_url()}/api/status", timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def run_pipeline(data_file: str, db_path: str | None, index_path: str | None):
    payload = {"data_file": data_file}
    if db_path:
        payload["db_path"] = db_path
    if index_path:
        payload["index_path"] = index_path
    try:
        r = httpx.post(f"{api_url()}/api/pipeline/run", json=payload, timeout=120.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e), "status": "failed"}


def similar_prompts(prompt_id: str, limit: int, threshold: float):
    try:
        r = httpx.get(
            f"{api_url()}/api/prompts/{prompt_id}/similar",
            params={"limit": limit, "threshold": threshold},
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def semantic_search(query: str, limit: int):
    try:
        r = httpx.post(
            f"{api_url()}/api/search/semantic",
            json={"query": query, "limit": limit},
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def duplicates(threshold: float, same_layer: bool, tier: str | None):
    params = {"threshold": threshold, "same_layer": same_layer}
    if tier:
        params["tier"] = tier
    try:
        r = httpx.get(f"{api_url()}/api/analysis/duplicates", params=params, timeout=30.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# --- Sidebar ---
with st.sidebar:
    st.title("Settings")
    st.session_state["api_url"] = st.text_input(
        "API base URL",
        value=DEFAULT_API_URL,
        help="FastAPI server URL (e.g. http://localhost:8000)",
    )
    st.divider()
    status = get_status()
    if "error" in status:
        st.error(f"API: {status['error']}")
    else:
        st.success("API connected")
        st.metric("Prompts", status.get("prompt_count", 0))
        st.metric("Embeddings", status.get("embedding_count", 0))

# --- Main: Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Pipeline",
    "Similar prompts",
    "Semantic search",
    "Duplicate analysis",
    "Visualizations",
])

with tab1:
    st.header("Run pipeline")
    st.caption("Load prompts from JSON and generate embeddings (idempotent).")
    col1, col2 = st.columns(2)
    with col1:
        data_file = st.text_input("Data file path", value="sample_prompts.json", key="data_file")
        db_path = st.text_input("Prompts DB path (optional)", value="", key="db_path")
    with col2:
        index_path = st.text_input("Embeddings DB path (optional)", value="", key="index_path")
    if st.button("Run pipeline"):
        with st.spinner("Loading prompts and generating embeddings…"):
            result = run_pipeline(
                data_file,
                db_path.strip() or None,
                index_path.strip() or None,
            )
        if "error" in result and "status" in result:
            st.error(result["error"])
        else:
            st.success("Pipeline finished")
            st.json(result)

with tab2:
    st.header("Similar prompts")
    st.caption("Find prompts most similar to a given prompt by embedding.")
    status = get_status()
    prompt_ids = []
    if "error" not in status and status.get("prompt_count", 0) > 0:
        try:
            r = httpx.get(f"{api_url()}/api/status", timeout=5.0)
            if r.ok:
                # Fetch prompt list from API by trying similar with limit=0 to get names, or we need an endpoint. Use a placeholder list or run similar for first id.
                # We don't have GET /api/prompts list. So use text input.
                pass
        except Exception:
            pass
    prompt_id = st.text_input("Prompt ID", value="survey.question.base", key="similar_id")
    limit = st.slider("Limit", 1, 50, 5, key="similar_limit")
    threshold = st.slider("Min similarity", 0.0, 1.0, 0.8, 0.05, key="similar_threshold")
    if st.button("Search similar", key="btn_similar"):
        out = similar_prompts(prompt_id, limit, threshold)
        if "error" in out:
            st.error(out["error"])
        else:
            st.subheader(f"Query: {out.get('query_prompt_id', '')}")
            if out.get("results"):
                st.dataframe(
                    [
                        {
                            "Prompt ID": r["prompt_id"],
                            "Similarity": r["similarity_score"],
                            "Layer": r["layer"],
                            "Category": r["category"],
                            "Preview": r["content_preview"][:80] + "…" if len(r["content_preview"]) > 80 else r["content_preview"],
                        }
                        for r in out["results"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No results above threshold.")

with tab3:
    st.header("Semantic search")
    st.caption("Search prompts by natural language query.")
    query = st.text_input("Query", value="how to greet a caller", key="semantic_query")
    limit = st.slider("Limit", 1, 50, 10, key="semantic_limit")
    if st.button("Search", key="btn_semantic"):
        out = semantic_search(query, limit)
        if "error" in out:
            st.error(out["error"])
        else:
            if out.get("results"):
                st.dataframe(
                    [
                        {
                            "Prompt ID": r["prompt_id"],
                            "Similarity": r["similarity_score"],
                            "Layer": r["layer"],
                            "Category": r["category"],
                            "Preview": r["content_preview"][:80] + "…" if len(r["content_preview"]) > 80 else r["content_preview"],
                        }
                        for r in out["results"]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No results.")

with tab4:
    st.header("Duplicate analysis")
    st.caption("Metadata-aware duplicate clusters (Tier1/2/3).")
    threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.9, 0.05, key="dup_threshold")
    same_layer = st.checkbox("Same layer only (exclude Tier3)", value=True, key="dup_same_layer")
    tier = st.selectbox("Filter tier", [None, "Tier1", "Tier2", "Tier3"], format_func=lambda x: "All" if x is None else x, key="dup_tier")
    if st.button("Get duplicates", key="btn_duplicates"):
        out = duplicates(threshold, same_layer, tier)
        if "error" in out:
            st.error(out["error"])
        else:
            st.metric("Total clusters", out.get("total_clusters", 0))
            st.json(out.get("tier_breakdown", {}))
            for dup in out.get("duplicates", []):
                with st.expander(f"{dup.get('cluster_id', '')} — {dup.get('tier', '')} ({dup.get('recommendation', '')})"):
                    st.write("**Target:**", dup.get("target_prompt_id"))
                    st.write("**Reason:**", dup.get("reason", ""))
                    st.write("**Merge candidates:**", dup.get("merge_candidates", []))
                    st.write("**Variable summary:**", dup.get("variable_summary", ""))
                    st.write("**Prompts:**", dup.get("prompts", []))

with tab5:
    st.header("Visualizations")
    st.caption("Generate and open D3 cluster graph, similarity heatmap, tier breakdown.")
    viz_dir = Path(__file__).parent / "visualizations"
    if st.button("Generate visualizations"):
        with st.spinner("Generating HTML…"):
            try:
                from visualize import generate_all
                generate_all()
                st.success(f"Generated in {viz_dir}")
            except Exception as e:
                st.error(str(e))
    if viz_dir.exists():
        index_html = viz_dir / "index.html"
        if index_html.exists():
            st.markdown(f"Open [visualizations/index.html](file://{index_html.resolve()}) in your browser.")
            st.write("Or run from project root: `python visualize.py` then open `visualizations/index.html`.")
