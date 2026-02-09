"""
Streamlit frontend for the LLM Response Quality Evaluator.

Run from project root (usecase_3): streamlit run frontend/app.py
Ensure the API is running: uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st

# Default API base URL (no /api suffix; we append /api/evaluate etc.)
DEFAULT_API_BASE = os.environ.get("EVALUATOR_API_BASE", "http://localhost:8000")

# Sample evaluation cases (from task description)
SAMPLE_CASES = [
    {
        "id": "eval_001",
        "context": {
            "directive": "Verify the caller's date of birth",
            "user_input": "It's March 15th, 1985",
        },
        "response_good": "Got it, March 15th, 1985. Thank you for confirming that.",
        "response_bad": "Perfect! I have recorded your date of birth as March 15th, 1985. Is there anything else I can help you verify today before we move on?",
        "expected_winner": "response_good",
        "reasoning": "Good response is concise and appropriate; bad response is verbose and goes beyond the task",
    },
    {
        "id": "eval_002",
        "context": {
            "directive": "Ask about food security using SDOH screening question",
            "user_input": "We haven't had enough to eat in weeks. It's been really hard.",
        },
        "response_good": "I'm sorry to hear that - that sounds really difficult. I want to make sure we can connect you with the right resources. In the past 12 months, have you or anyone in your household run out of food before you could get more?",
        "response_bad": "Okay, noted. Next question: Within the past 12 months, did you worry that your food would run out before you got money to buy more?",
        "expected_winner": "response_good",
        "reasoning": "Good response shows empathy before continuing; bad response ignores emotional content",
    },
    {
        "id": "eval_003",
        "context": {
            "directive": "Handle user confusion about the survey purpose",
            "user_input": "Wait, why are you asking me all these personal questions? Who is this for?",
        },
        "response_good": "That's a fair question. This survey helps us understand if there are any areas where you might benefit from community resources - like food, housing, or support. Your answers stay private and help us point you to the right help.",
        "response_bad": "I'm an AI assistant helping with your health screening. These questions are part of a standard SDOH assessment protocol that we're required to complete.",
        "expected_winner": "response_good",
        "reasoning": "Good response is reassuring and human; bad response is cold and bureaucratic",
    },
]


def api_headers(token: str | None) -> dict[str, str]:
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _request_error(e: requests.exceptions.RequestException) -> str:
    if hasattr(e, "response") and e.response is not None and e.response.text:
        try:
            detail = e.response.json()
            return str(detail.get("detail", e))
        except Exception:
            pass
    return str(e)


def post_evaluate(base: str, token: str | None, body: dict) -> tuple[dict | None, str]:
    try:
        r = requests.post(
            f"{base}/api/evaluate",
            json=body,
            headers=api_headers(token),
            timeout=30,
        )
        r.raise_for_status()
        return r.json(), ""
    except requests.exceptions.RequestException as e:
        return None, _request_error(e)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"


def post_compare(base: str, token: str | None, body: dict) -> tuple[dict | None, str]:
    try:
        r = requests.post(f"{base}/api/compare", json=body, headers=api_headers(token), timeout=60)
        r.raise_for_status()
        return r.json(), ""
    except requests.exceptions.RequestException as e:
        return None, _request_error(e)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"


def post_improve(base: str, token: str | None, body: dict) -> tuple[dict | None, str]:
    try:
        r = requests.post(f"{base}/api/improve", json=body, headers=api_headers(token), timeout=60)
        r.raise_for_status()
        return r.json(), ""
    except requests.exceptions.RequestException as e:
        return None, _request_error(e)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"


def post_batch(base: str, token: str | None, body: dict) -> tuple[dict | None, str]:
    try:
        r = requests.post(f"{base}/api/evaluate/batch", json=body, headers=api_headers(token), timeout=120)
        r.raise_for_status()
        return r.json(), ""
    except requests.exceptions.RequestException as e:
        return None, _request_error(e)
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"


def build_context(directive: str, user_input: str, conversation_history: list[dict] | None) -> dict:
    ctx = {"directive": directive or None, "user_input": user_input or None}
    if conversation_history:
        ctx["conversation_history"] = conversation_history
    return ctx


def main() -> None:
    st.set_page_config(page_title="LLM Response Evaluator", layout="wide")
    st.title("LLM Response Quality Evaluator")
    st.caption("Score responses, compare A/B variants, flag issues, and get improvement suggestions.")

    with st.sidebar:
        api_base = st.text_input("API base URL", value=DEFAULT_API_BASE, help="e.g. http://localhost:8000")
        bearer_token = st.text_input("Bearer token (optional)", type="password", help="Set if EVAL_REQUIRE_AUTH=1")
        st.divider()
        st.markdown("**Endpoints**")
        st.markdown("- POST /api/evaluate")
        st.markdown("- POST /api/evaluate/batch")
        st.markdown("- POST /api/compare")
        st.markdown("- POST /api/improve")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Evaluate single",
        "Batch evaluate",
        "Compare A/B",
        "Improve response",
        "Sample cases",
    ])

    # ---- Evaluate single ----
    with tab1:
        st.subheader("Evaluate a single response")
        _eval_directive = "Collect whether the member is currently employed (yes/no)."
        _eval_user_input = "I'm between jobs right now."
        _eval_response = "So right now you're not employed—we'll mark that as no. Is there anything else about work you want to add?"
        _eval_conv = json.dumps([
            {"role": "user", "content": "I'm calling about my benefits."},
            {"role": "assistant", "content": "I can help with that. Are you currently employed, yes or no?"},
            {"role": "user", "content": "I'm between jobs right now."},
        ], indent=2)
        directive = st.text_input("Directive", value=_eval_directive, key="eval_directive")
        user_input = st.text_area("User input (current turn)", value=_eval_user_input, height=60, key="eval_user_input")
        response = st.text_area("AI response to evaluate", value=_eval_response, height=120, key="eval_response")
        col1, col2, col3 = st.columns(3)
        with col1:
            context_type = st.selectbox("Context type", ["verification", "screening", "clarification", "follow_up"], index=0, key="eval_ctx")
        with col2:
            agent_id = st.selectbox("Agent ID", ["survey_agent", "screening_agent", "verification_agent", "clarification_agent", "support_agent"], index=2, key="eval_agent")
        with col3:
            prompt_version = st.text_input("Prompt version", value="v2.1", key="eval_pv")
        conv_json = st.text_area("Conversation history (JSON array, optional)", value=_eval_conv, height=120, key="eval_conv")
        if st.button("Evaluate", key="btn_eval"):
            if not response.strip():
                st.error("Please enter a response to evaluate.")
            else:
                try:
                    conv = json.loads(conv_json) if conv_json.strip() else []
                except json.JSONDecodeError:
                    st.error("Invalid JSON in conversation history.")
                else:
                    # API expects conversation_history as { "turns": [ { "role", "content" }, ... ] } with 3–20 turns
                    if isinstance(conv, list) and len(conv) >= 3:
                        conv_payload = {"turns": [{"role": t.get("role", "user"), "content": (t.get("content") or "").strip() or " "} for t in conv]}
                    else:
                        conv_payload = None
                    body = {
                        "response": response.strip(),
                        "directive": directive.strip() or None,
                        "user_input": user_input.strip() or None,
                        "conversation_history": conv_payload,
                        "metadata": {"context_type": context_type, "agent_id": agent_id, "prompt_version": prompt_version},
                    }
                    result, err = post_evaluate(api_base, bearer_token or None, body)
                    if err:
                        st.error(err)
                    else:
                        st.success("Evaluation complete")
                        st.metric("Overall score", result.get("overall_score"))
                        st.json(result.get("dimensions", {}))
                        if result.get("flags"):
                            st.warning("Flags: " + ", ".join(f.get("code", "") for f in result["flags"]))
                        if result.get("suggestions"):
                            st.info("Suggestions: " + " | ".join(result["suggestions"]))

    # ---- Batch ----
    with tab2:
        st.subheader("Batch evaluate")
        _batch_placeholder = [
            {"response": "So right now you're not employed—we'll mark that as no. Is there anything else about work you want to add?", "directive": "Collect whether the member is currently employed (yes/no).", "user_input": "I'm between jobs right now.", "metadata": {"context_type": "verification", "agent_id": "verification_agent", "prompt_version": "v2.1"}},
            {"response": "I hear you, that can be really hard. One more question we ask everyone—in the last 12 months, have you or anyone in your household run out of food before you could get more?", "directive": "Ask the standard food security screening question clearly; allow user to answer.", "user_input": "I've been really stressed about bills.", "metadata": {"context_type": "screening", "agent_id": "screening_agent", "prompt_version": "v2.1"}},
            {"response": "Okay, noted. Next question: Within the past 12 months, did you worry that your food would run out before you got money to buy more?", "directive": "Ask the standard food security screening question clearly.", "user_input": "We haven't had enough to eat in weeks. It's been really hard.", "metadata": {"context_type": "screening", "agent_id": "screening_agent", "prompt_version": "v2.1"}},
        ]
        batch_json = st.text_area("JSON array of evaluations", value=json.dumps(_batch_placeholder, indent=2), height=280, key="batch_json")
        if st.button("Run batch", key="btn_batch"):
            if not batch_json.strip():
                st.error("Paste or upload a JSON array of evaluation requests.")
            else:
                try:
                    evals_list = json.loads(batch_json)
                    if not isinstance(evals_list, list):
                        st.error("Root must be a JSON array.")
                    else:
                        body = {"evaluations": evals_list[:500]}
                        result, err = post_batch(api_base, bearer_token or None, body)
                        if err:
                            st.error(err)
                        else:
                            st.success("Batch complete")
                            st.metric("Evaluated", result.get("metadata", {}).get("total_evaluated", 0))
                            st.metric("Cache hit rate", f"{result.get('metadata', {}).get('cache_hit_rate', 0)*100:.1f}%")
                            if "aggregate_stats" in result:
                                st.subheader("Aggregate stats")
                                st.json(result["aggregate_stats"])
                            if "individual_scores" in result:
                                st.dataframe([{"eval_id": s.get("eval_id"), "overall": s.get("overall_score")} for s in result["individual_scores"][:50]])
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")

    # ---- Compare A/B ----
    with tab3:
        st.subheader("Compare two responses (A/B)")
        _cmp_directive = "Ask about food security using SDOH screening question"
        _cmp_user_input = "We haven't had enough to eat in weeks. It's been really hard."
        _cmp_a = "I'm sorry to hear that - that sounds really difficult. I want to make sure we can connect you with the right resources. In the past 12 months, have you or anyone in your household run out of food before you could get more?"
        _cmp_b = "Okay, noted. Next question: Within the past 12 months, did you worry that your food would run out before you got money to buy more?"
        cmp_directive = st.text_input("Directive", value=_cmp_directive, key="cmp_directive")
        cmp_user_input = st.text_input("User input", value=_cmp_user_input, key="cmp_user_input")
        response_a = st.text_area("Response A (e.g. good)", value=_cmp_a, height=100, key="cmp_a")
        response_b = st.text_area("Response B (e.g. bad)", value=_cmp_b, height=100, key="cmp_b")
        if st.button("Compare", key="btn_compare"):
            if not response_a.strip() or not response_b.strip():
                st.error("Enter both Response A and Response B.")
            else:
                body = {
                    "context": build_context(cmp_directive, cmp_user_input, None),
                    "response_a": response_a.strip(),
                    "response_b": response_b.strip(),
                }
                result, err = post_compare(api_base, bearer_token or None, body)
                if err:
                    st.error(err)
                else:
                    winner = result.get("winner", "tie")
                    st.success(f"Winner: **{winner.upper()}**")
                    st.metric("Confidence in winner", f"{result.get('confidence_in_winner', 0):.2f}")
                    st.write("Recommendation:", result.get("recommendation", ""))
                    if result.get("dimension_comparisons"):
                        st.subheader("Per dimension")
                        for dc in result["dimension_comparisons"]:
                            st.write(f"- **{dc.get('dimension')}**: winner={dc.get('winner')}, A={dc.get('score_a')}, B={dc.get('score_b')}")

    # ---- Improve ----
    with tab4:
        st.subheader("Improve a low-scoring response")
        _imp_directive = "Ask the standard food security screening question clearly; allow user to answer."
        _imp_user_input = "We haven't had enough to eat in weeks. It's been really hard."
        _imp_response = "Okay, noted. Next question: Within the past 12 months, did you worry that your food would run out before you got money to buy more?"
        imp_directive = st.text_input("Directive", value=_imp_directive, key="imp_directive")
        imp_user_input = st.text_input("User input", value=_imp_user_input, key="imp_user_input")
        imp_response = st.text_area("Original response", value=_imp_response, height=120, key="imp_response")
        target_dims = st.multiselect("Target dimensions", ["empathy", "conciseness", "naturalness", "clarity", "task_completion", "safety"], default=["empathy"], key="imp_dims")
        if st.button("Get improvement", key="btn_improve"):
            if not imp_response.strip():
                st.error("Enter the response to improve.")
            else:
                body = {
                    "response": imp_response.strip(),
                    "context": build_context(imp_directive, imp_user_input, None),
                    "target_dimensions": target_dims or None,
                    "num_variants": 1,
                }
                result, err = post_improve(api_base, bearer_token or None, body)
                if err:
                    st.error(err)
                else:
                    st.success("Improvement generated")
                    st.metric("Confidence in improvement", f"{result.get('confidence_in_improvement', 0):.2f}")
                    st.write("**Original score (overall):**", result.get("original_scores", {}).get("overall"))
                    st.write("**Predicted score (overall):**", result.get("predicted_scores", {}).get("overall"))
                    st.text_area("Improved response", value=result.get("improved_response", ""), height=150, disabled=True)
                    if result.get("changes_made"):
                        st.write("**Changes made:**")
                        for c in result["changes_made"]:
                            st.write(f"- [{c.get('category')}] {c.get('description')}")

    # ---- Sample cases ----
    with tab5:
        st.subheader("Sample evaluation cases (good vs bad)")
        st.caption("Run A/B compare on the three sample cases; expected winner is the 'good' response.")
        for case in SAMPLE_CASES:
            with st.expander(f"**{case['id']}** — {case['context']['directive'][:50]}..."):
                st.write("**Expected winner:**", case["expected_winner"])
                st.write("**Reasoning:**", case["reasoning"])
                if st.button(f"Run compare for {case['id']}", key=f"run_{case['id']}"):
                    body = {
                        "context": {"directive": case["context"]["directive"], "user_input": case["context"]["user_input"]},
                        "response_a": case["response_good"],
                        "response_b": case["response_bad"],
                    }
                    result, err = post_compare(api_base, bearer_token or None, body)
                    if err:
                        st.error(err)
                    else:
                        actual = result.get("winner", "tie")
                        expected_a = "a" if case["expected_winner"] == "response_good" else "b"
                        match = "✓ Match" if actual == expected_a else "✗ Mismatch"
                        st.success(f"Winner: **{actual}** — {match}")
                        st.write("Confidence:", result.get("confidence_in_winner"))
                        st.write("Recommendation:", result.get("recommendation", ""))


if __name__ == "__main__":
    main()
