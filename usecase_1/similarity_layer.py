"""
Prompt Similarity Service — Similarity & Clustering Engine.

Metadata-aware deduplication:
- Tier1: same_layer + same_category + sim >= 0.92 → recommend merge
- Tier2: same_layer + different_category + sim >= 0.90 → flag for review
- Tier3: different_layer + sim >= 0.88 → informational only
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cosine, squareform

# Type aliases
Tier = Literal["Tier1", "Tier2", "Tier3", "NoMatch"]
Recommendation = Literal["MERGE", "REVIEW", "KEEP_SEPARATE"]

# Tier thresholds (similarity)
TIER1_THRESHOLD = 0.92
TIER2_THRESHOLD = 0.90
TIER3_THRESHOLD = 0.88

# Variable placeholders: {{x}} or [VARIABLE_X]
VARIABLE_BRACE = re.compile(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}")
VARIABLE_ANCHOR = re.compile(r"\[VARIABLE_([A-Z0-9_]+)\]")


def _extract_variable_names(content: str) -> set[str]:
    """Return variable names (snake_case) for summary (from raw {{x}} or normalized [VARIABLE_X])."""
    names = set()
    for m in VARIABLE_BRACE.finditer(content):
        names.add(m.group(1))
    for m in VARIABLE_ANCHOR.finditer(content):
        names.add(m.group(1).lower())
    return names


class SimilarityComputer:
    """Cosine similarity in [0, 1]. Handles zero vectors and NaN."""

    @staticmethod
    def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Returns cosine similarity in [0.0, 1.0].
        Uses scipy cosine distance: similarity = 1 - distance.
        """
        a = np.asarray(embedding1, dtype=np.float64).ravel()
        b = np.asarray(embedding2, dtype=np.float64).ravel()
        if a.shape != b.shape:
            raise ValueError("Embeddings must have the same dimension")
        if a.size == 0:
            return 0.0
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        dist = cosine(a, b)
        sim = 1.0 - float(dist)
        return max(0.0, min(1.0, sim))


class MetadataAwareMatcher:
    """Applies tier rules to pairwise similarities using prompt metadata."""

    def __init__(self, similarity_threshold: float = 0.88) -> None:
        self.similarity_threshold = similarity_threshold

    def compute_pairwise_similarities(
        self,
        prompts: List["PromptRecord"],
        embeddings: Dict[str, np.ndarray],
    ) -> Dict[Tuple[str, str], float]:
        """
        Returns dict ((id1, id2), similarity) for upper triangle only (id1 < id2).
        """
        from data_layer import PromptRecord  # noqa: F401

        ids = [p.prompt_id for p in prompts]
        by_id = {p.prompt_id: p for p in prompts}
        computer = SimilarityComputer()
        out: Dict[Tuple[str, str], float] = {}
        for i in range(len(ids)):
            id_i = ids[i]
            if id_i not in embeddings:
                continue
            emb_i = embeddings[id_i]
            for j in range(i + 1, len(ids)):
                id_j = ids[j]
                if id_j not in embeddings:
                    continue
                sim = computer.compute_similarity(emb_i, embeddings[id_j])
                key = (id_i, id_j) if id_i < id_j else (id_j, id_i)
                out[key] = sim
        return out

    def apply_metadata_filter(
        self,
        pair: Tuple["PromptRecord", "PromptRecord"],
        similarity: float,
    ) -> Tuple[Tier, float]:
        """
        Returns (tier, confidence_score). confidence_score is 0.0 for NoMatch.
        """
        from data_layer import PromptRecord  # noqa: F401

        a, b = pair
        same_layer = a.layer == b.layer
        same_category = a.category == b.category

        if same_layer and same_category and similarity >= TIER1_THRESHOLD:
            return ("Tier1", similarity)
        if same_layer and not same_category and similarity >= TIER2_THRESHOLD:
            return ("Tier2", similarity)
        if not same_layer and similarity >= TIER3_THRESHOLD:
            return ("Tier3", similarity)
        return ("NoMatch", 0.0)


class DuplicateClusterer:
    """
    Builds clusters per tier using hierarchical clustering (complete linkage).
    Input: similarities dict (id1, id2) -> (tier, score) for pairs that passed metadata filter.
    """

    def __init__(
        self,
        tier1_threshold: float = TIER1_THRESHOLD,
        tier2_threshold: float = TIER2_THRESHOLD,
        tier3_threshold: float = TIER3_THRESHOLD,
    ) -> None:
        self.tier1_threshold = tier1_threshold
        self.tier2_threshold = tier2_threshold
        self.tier3_threshold = tier3_threshold

    def cluster_by_tier(
        self,
        similarities: Dict[Tuple[str, str], Tuple[Tier, float]],
        min_cluster_size: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        similarities: (id1, id2) -> (tier, score). Only include pairs with tier in Tier1, Tier2, Tier3.
        Returns list of Cluster dicts, sorted by confidence descending.
        """
        tier_reasons = {
            "Tier1": "Same layer + category + high semantic similarity",
            "Tier2": "Same layer, different category — review recommended",
            "Tier3": "Different layer — informational only",
        }
        tier_thresholds: Dict[Tier, float] = {
            "Tier1": self.tier1_threshold,
            "Tier2": self.tier2_threshold,
            "Tier3": self.tier3_threshold,
        }
        all_clusters: List[Dict[str, Any]] = []
        cluster_counter = [0]

        def next_id() -> str:
            cluster_counter[0] += 1
            return f"dup_{cluster_counter[0]:03d}"

        for tier in ("Tier1", "Tier2", "Tier3"):
            pairs_in_tier = [
                (k, v[1])
                for k, v in similarities.items()
                if v[0] == tier and v[1] >= tier_thresholds[tier]
            ]
            if not pairs_in_tier:
                continue
            ids = sorted(set(p for k in [k for k, _ in pairs_in_tier] for p in k))
            id_to_idx = {pid: i for i, pid in enumerate(ids)}
            n = len(ids)
            if n < 2:
                continue
            t_cut = 1.0 - tier_thresholds[tier]
            # Full distance matrix: 1 - similarity; unknown pairs = t_cut so transitive pairs merge
            dist = np.full((n, n), t_cut, dtype=np.float64)
            np.fill_diagonal(dist, 0)
            for (id1, id2), sim in pairs_in_tier:
                i, j = id_to_idx[id1], id_to_idx[id2]
                d = 1.0 - sim
                dist[i, j] = dist[j, i] = min(dist[i, j], d)
            condensed = squareform(dist, checks=False)
            try:
                Z = linkage(condensed, method="complete")
            except Exception:
                continue
            labels = fcluster(Z, t=t_cut, criterion="distance")
            # Build clusters: each label is a cluster
            label_to_ids: Dict[int, List[str]] = {}
            for idx, label in enumerate(labels):
                label_to_ids.setdefault(int(label), []).append(ids[idx])
            for label, member_ids in label_to_ids.items():
                if len(member_ids) < min_cluster_size:
                    continue
                # Pairwise similarities within this cluster (from our tier pairs)
                pair_sims = {}
                for (id1, id2), sim in pairs_in_tier:
                    if id1 in member_ids and id2 in member_ids:
                        pair_sims[(id1, id2)] = sim
                # Per-prompt max similarity to another in cluster
                prompt_sims: Dict[str, float] = {pid: 0.0 for pid in member_ids}
                for (id1, id2), sim in pair_sims.items():
                    prompt_sims[id1] = max(prompt_sims[id1], sim)
                    prompt_sims[id2] = max(prompt_sims[id2], sim)
                # If a prompt never appeared in a pair, use max of pair_sims or 0
                if pair_sims:
                    default_sim = max(pair_sims.values())
                    for pid in member_ids:
                        if prompt_sims[pid] == 0:
                            prompt_sims[pid] = default_sim
                confidence = max(prompt_sims.values()) if prompt_sims else 0.0
                all_clusters.append({
                    "cluster_id": next_id(),
                    "tier": tier,
                    "prompts": [{"prompt_id": pid, "similarity": round(prompt_sims[pid], 4)} for pid in member_ids],
                    "confidence": round(confidence, 4),
                    "reason": tier_reasons[tier],
                })
        all_clusters.sort(key=lambda c: c["confidence"], reverse=True)
        return all_clusters


class MergeRecommendationBuilder:
    """Builds merge recommendations from clusters. Canonical = longest content."""

    def suggest_merge(
        self,
        cluster: Dict[str, Any],
        prompts: Dict[str, "PromptRecord"],
    ) -> Dict[str, Any]:
        """
        Returns MergeRecommendation dict.
        target_prompt_id = canonical (longest content); merge_candidates = others.
        """
        from data_layer import PromptRecord  # noqa: F401

        tier = cluster["tier"]
        if tier == "Tier1":
            recommendation: Recommendation = "MERGE"
        elif tier == "Tier2":
            recommendation = "REVIEW"
        else:
            recommendation = "KEEP_SEPARATE"

        member_ids = [p["prompt_id"] for p in cluster["prompts"]]
        records = [prompts[pid] for pid in member_ids if pid in prompts]
        if not records:
            return {
                "cluster_id": cluster["cluster_id"],
                "recommendation": recommendation,
                "target_prompt_id": member_ids[0],
                "merge_candidates": member_ids[1:],
                "reason": cluster["reason"],
                "variable_summary": "",
                "confidence": cluster["confidence"],
            }
        canonical = max(records, key=lambda r: len(r.content))
        target_id = canonical.prompt_id
        candidates = [pid for pid in member_ids if pid != target_id]

        # Variable summary: union of {{var}} from all prompts in cluster
        all_vars: set[str] = set()
        for r in records:
            all_vars |= _extract_variable_names(r.content)
        vars_str = ", ".join(f"{{{{{v}}}}}" for v in sorted(all_vars))
        prefix = "Both use " if len(member_ids) > 1 else "Uses "
        var_summary = (prefix + vars_str) if all_vars else "No template variables"

        return {
            "cluster_id": cluster["cluster_id"],
            "recommendation": recommendation,
            "target_prompt_id": target_id,
            "merge_candidates": candidates,
            "reason": cluster["reason"],
            "variable_summary": var_summary,
            "confidence": cluster["confidence"],
        }
