"""
Prompt Similarity Service — Visualization tools.

Generates static HTML: D3.js force-directed cluster graph, Plotly similarity heatmap,
and tier breakdown bar chart. All outputs go to visualizations/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from data_layer import PromptStore
from embedding_layer import PromptEmbeddingStore, PromptNormalizer
from similarity_layer import (
    DuplicateClusterer,
    MetadataAwareMatcher,
    SimilarityComputer,
)

VISUALIZATIONS_DIR = Path(__file__).parent / "visualizations"
TIER_COLORS = {"Tier1": "#e74c3c", "Tier2": "#e67e22", "Tier3": "#95a5a6"}
TIER_ORDER = ("Tier1", "Tier2", "Tier3")


def _load_data(
    db_path: str,
    index_path: str,
    threshold: float = 0.85,
) -> Tuple[List[Any], Dict[str, np.ndarray], List[Dict], Dict[Tuple[str, str], float], Dict[str, str]]:
    """Load prompts, embeddings, clusters, pairwise sims, and prompt_id -> tier (for nodes)."""
    store = PromptStore()
    store.open_db(db_path)
    emb_store = PromptEmbeddingStore(index_path)
    prompts = store.get_all_prompts()
    embeddings = emb_store.load_all_embeddings()
    if not prompts or not embeddings:
        return prompts, embeddings, [], {}, {}
    matcher = MetadataAwareMatcher()
    sims_raw = matcher.compute_pairwise_similarities(prompts, embeddings)
    by_id = {p.prompt_id: p for p in prompts}
    filtered: Dict[Tuple[str, str], Tuple[str, float]] = {}
    for (id1, id2), sim in sims_raw.items():
        if sim < threshold:
            continue
        t, conf = matcher.apply_metadata_filter((by_id[id1], by_id[id2]), sim)
        if t == "NoMatch":
            continue
        filtered[(id1, id2)] = (t, conf)
    clusterer = DuplicateClusterer(
        tier1_threshold=max(threshold, 0.92),
        tier2_threshold=max(threshold, 0.90),
        tier3_threshold=max(threshold, 0.88),
    )
    clusters = clusterer.cluster_by_tier(filtered, min_cluster_size=2)
    # prompt_id -> best tier (Tier1 > Tier2 > Tier3)
    pid_to_tier: Dict[str, str] = {}
    for c in clusters:
        t = c["tier"]
        for p in c["prompts"]:
            pid = p["prompt_id"]
            if pid not in pid_to_tier or TIER_ORDER.index(t) < TIER_ORDER.index(pid_to_tier[pid]):
                pid_to_tier[pid] = t
    return prompts, embeddings, clusters, sims_raw, pid_to_tier


def cluster_visualization(
    db_path: str = "prompts.db",
    index_path: str = "embeddings.db",
    threshold: float = 0.85,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate HTML with D3.js force-directed graph.
    Nodes = prompts (colored by tier). Edges = similarity links (thickness = score).
    Interactive: hover shows prompt_id, click shows full content.
    """
    out = output_path or VISUALIZATIONS_DIR / "clusters.html"
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    prompts, embeddings, clusters, sims_raw, pid_to_tier = _load_data(db_path, index_path, threshold)
    by_id = {p.prompt_id: p for p in prompts}
    # Nodes: all prompts that have embeddings (or only those in clusters for smaller graph)
    ids = sorted(embeddings.keys()) if embeddings else []
    node_tiers = [pid_to_tier.get(pid, "None") for pid in ids]
    node_colors = [TIER_COLORS.get(t, "#bdc3c7") for t in node_tiers]
    # Edges: from sims_raw, only above threshold for display
    links: List[Dict[str, Any]] = []
    seen_pairs = set()
    for (id1, id2), sim in sims_raw.items():
        if sim < threshold or (id1, id2) in seen_pairs or (id2, id1) in seen_pairs:
            continue
        seen_pairs.add((id1, id2))
        links.append({"source": id1, "target": id2, "similarity": round(sim, 3)})
    # Limit edges for readability (top 50 by similarity)
    links.sort(key=lambda x: x["similarity"], reverse=True)
    links = links[:50]
    nodes_data = [
        {"id": pid, "tier": pid_to_tier.get(pid, "None"), "content": (by_id[pid].content[:200] if pid in by_id else "")}
        for pid in ids
    ]
    nodes_json = json.dumps(nodes_data)
    links_json = json.dumps(links)
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Prompt similarity clusters</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    body {{ font-family: sans-serif; margin: 20px; }}
    .legend {{ margin-bottom: 16px; }}
    .legend span {{ margin-right: 16px; }}
    #graph {{ width: 100%; height: 600px; border: 1px solid #ccc; }}
    .node {{ cursor: pointer; }}
    .node:hover {{ stroke: #333; stroke-width: 2px; }}
    .link {{ stroke-opacity: 0.6; }}
    .tooltip {{ position: absolute; padding: 8px; background: #fff; border: 1px solid #ccc; border-radius: 4px; pointer-events: none; max-width: 400px; font-size: 12px; }}
  </style>
</head>
<body>
  <h1>Prompt similarity clusters</h1>
  <div class="legend">
    <strong>Legend:</strong>
    <span style="color:#e74c3c">Red = Tier1 (merge)</span>
    <span style="color:#e67e22">Orange = Tier2 (review)</span>
    <span style="color:#95a5a6">Gray = Tier3 (info)</span>
    <span style="color:#bdc3c7">Light gray = No cluster</span>
  </div>
  <div id="graph"></div>
  <div id="tooltip" class="tooltip" style="display:none;"></div>
  <script>
    const nodesData = {nodes_json};
    const linksData = {links_json};
    const width = document.getElementById("graph").clientWidth || 800;
    const height = 600;
    const idToIndex = {{}};
    nodesData.forEach((d, i) => idToIndex[d.id] = i);
    const nodes = nodesData.map((d, i) => ({{ ...d, index: i }}));
    const links = linksData.map(l => ({{
      source: idToIndex[l.source] ?? 0,
      target: idToIndex[l.target] ?? 0,
      similarity: l.similarity
    }}));
    const simulation = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.index).distance(100))
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(width/2, height/2));
    const svg = d3.select("#graph").append("svg").attr("width", width).attr("height", height);
    const g = svg.append("g");
    const link = g.append("g").selectAll("line")
      .data(links)
      .join("line")
      .attr("class", "link")
      .attr("stroke-width", d => Math.max(1, d.similarity * 4));
    const node = g.append("g").selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("class", "node")
      .attr("r", 8)
      .attr("fill", d => {{
        const colors = {{ "Tier1": "#e74c3c", "Tier2": "#e67e22", "Tier3": "#95a5a6", "None": "#bdc3c7" }};
        return colors[d.tier] || "#bdc3c7";
      }})
      .attr("stroke", "#333")
      .attr("stroke-width", 1)
      .on("mouseover", (e, d) => {{
        const tip = document.getElementById("tooltip");
        tip.style.display = "block";
        tip.style.left = (e.pageX + 10) + "px";
        tip.style.top = (e.pageY + 10) + "px";
        tip.innerHTML = "<b>" + d.id + "</b><br/>Tier: " + d.tier;
      }})
      .on("mousemove", (e) => {{
        document.getElementById("tooltip").style.left = (e.pageX + 10) + "px";
        document.getElementById("tooltip").style.top = (e.pageY + 10) + "px";
      }})
      .on("mouseout", () => {{ document.getElementById("tooltip").style.display = "none"; }})
      .on("click", (e, d) => {{
        const tip = document.getElementById("tooltip");
        tip.style.display = "block";
        tip.innerHTML = "<b>" + d.id + "</b><br/>" + (d.content || "").replace(/</g, "&lt;");
      }})
      .call(d3.drag()
        .on("start", (e, d) => {{ e.subject.fx = d.x; e.subject.fy = d.y; }})
        .on("drag", (e, d) => {{ e.subject.fx = e.x; e.subject.fy = e.y; }})
        .on("end", (e, d) => {{ e.subject.fx = null; e.subject.fy = null; }}));
    simulation.on("tick", () => {{
      link.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
          .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
      node.attr("cx", d => d.x).attr("cy", d => d.y);
    }});
  </script>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    return out


def similarity_matrix_heatmap(
    db_path: str = "prompts.db",
    index_path: str = "embeddings.db",
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate heatmap of pairwise similarity scores.
    Prompts ordered by (layer, category). Color gradient white (0.0) → red (1.0).
    Include dendrogram from hierarchical clustering.
    """
    out = output_path or VISUALIZATIONS_DIR / "similarity_matrix.html"
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    store = PromptStore()
    store.open_db(db_path)
    emb_store = PromptEmbeddingStore(index_path)
    prompts = store.get_all_prompts()
    embeddings = emb_store.load_all_embeddings()
    if not prompts or not embeddings:
        # Empty heatmap
        fig = go.Figure(data=go.Heatmap(z=[[0]], x=[""], y=[""], colorscale="Reds", zmin=0, zmax=1))
        fig.write_html(str(out))
        return out
    by_id = {p.prompt_id: p for p in prompts}
    ids = sorted(embeddings.keys())
    # Order by (layer, category)
    ids_ordered = sorted(ids, key=lambda i: (by_id[i].layer, by_id[i].category) if i in by_id else ("", ""))
    computer = SimilarityComputer()
    n = len(ids_ordered)
    matrix = np.zeros((n, n))
    for i, a in enumerate(ids_ordered):
        for j, b in enumerate(ids_ordered):
            if a in embeddings and b in embeddings:
                matrix[i, j] = computer.compute_similarity(embeddings[a], embeddings[b])
            else:
                matrix[i, j] = 0.0
    # Symmetry check: matrix[i][j] == matrix[j][i]
    assert np.allclose(matrix, matrix.T), "Similarity matrix must be symmetric"
    fig_heat = go.Figure()
    fig_heat.add_trace(go.Heatmap(
        z=matrix,
        x=ids_ordered,
        y=ids_ordered,
        colorscale="Reds",
        zmin=0,
        zmax=1,
        hoverongaps=False,
    ))
    fig_heat.update_layout(
        title="Pairwise similarity (ordered by layer, category)",
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
        width=900,
        height=700,
    )
    fig_heat.write_html(str(out))
    return out


def tier_breakdown_chart(
    db_path: str = "prompts.db",
    index_path: str = "embeddings.db",
    threshold: float = 0.85,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate bar chart: Y = number of clusters, X = Tier1, Tier2, Tier3.
    Tooltip shows details on hover.
    """
    out = output_path or VISUALIZATIONS_DIR / "tier_breakdown.html"
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    _, _, clusters, _, _ = _load_data(db_path, index_path, threshold)
    tier_counts = {"Tier1": 0, "Tier2": 0, "Tier3": 0}
    tier_details: Dict[str, List[str]] = {"Tier1": [], "Tier2": [], "Tier3": []}
    for c in clusters:
        t = c["tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1
        tier_details[t].append(f"{c['cluster_id']} (conf: {c['confidence']})")
    x = ["Tier1", "Tier2", "Tier3"]
    y = [tier_counts[t] for t in x]
    hover = [("<br>").join(tier_details[t]) if tier_details[t] else "No clusters" for t in x]
    fig = go.Figure(data=[go.Bar(x=x, y=y, text=y, textposition="auto", hovertext=hover, hoverinfo="text")])
    fig.update_layout(
        title="Cluster distribution by tier",
        xaxis_title="Tier",
        yaxis_title="Number of clusters",
        width=500,
        height=400,
    )
    fig.write_html(str(out))
    return out


def generate_all(
    db_path: str = "prompts.db",
    index_path: str = "embeddings.db",
    threshold: float = 0.85,
) -> None:
    """Generate all visualizations and index.html."""
    VISUALIZATIONS_DIR.mkdir(parents=True, exist_ok=True)
    cluster_visualization(db_path, index_path, threshold)
    similarity_matrix_heatmap(db_path, index_path)
    tier_breakdown_chart(db_path, index_path, threshold)
    index_html = VISUALIZATIONS_DIR / "index.html"
    index_html.write_text("""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Prompt Similarity Service — Visualizations</title>
</head>
<body>
  <h1>Prompt Similarity Service — Visualizations</h1>
  <ul>
    <li><a href="clusters.html">Cluster graph (D3 force-directed)</a></li>
    <li><a href="similarity_matrix.html">Similarity matrix heatmap</a></li>
    <li><a href="tier_breakdown.html">Tier breakdown bar chart</a></li>
  </ul>
</body>
</html>
""", encoding="utf-8")
    print(f"Generated visualizations in {VISUALIZATIONS_DIR}")


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "prompts.db"
    idx = sys.argv[2] if len(sys.argv) > 2 else "embeddings.db"
    generate_all(db, idx)
