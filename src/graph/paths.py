"""Structural IP graph and path search.

Two clearly separated objects:

``build_ip_graph``
    A compact undirected graph whose nodes are IP addresses and whose edges are
    *observed* communications aggregated from flow records. No labels are used.
    This is the graph the structural features are computed on, and it stays small
    (hundreds to a few thousand nodes even for 10^5 records).

``paths_to_intel``
    Breadth-first search from the set of IPs that carry training-derived
    threat-intel evidence. Used both as a feature and as supporting evidence in
    an alert explanation.

A path is *supporting context*, not proof: it says "the investigated entity is
reachable from an entity with historical malicious labels through observed
traffic", nothing more.
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.config import Settings, get_logger
from src.data.preprocess import is_private_ip

logger = get_logger("graph.paths")


def first_existing_column(df: pd.DataFrame, names: tuple[str, ...] | list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def resolve_entity_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Find the entity columns, tolerating alternative naming."""
    return {
        "src_ip": first_existing_column(df, ["Source IP", "src_ip", "source_ip"]),
        "dst_ip": first_existing_column(df, ["Destination IP", "dst_ip", "destination_ip"]),
        "src_port": first_existing_column(df, ["Source Port", "src_port"]),
        "dst_port": first_existing_column(df, ["Destination Port", "dst_port"]),
        "protocol": first_existing_column(df, ["Protocol", "protocol"]),
    }


def build_ip_graph(
    events: pd.DataFrame,
    settings: Settings,
    k_confidence: float | None = None,
) -> tuple[nx.Graph, dict[str, Any]]:
    """Aggregate observed flows into an IP-level graph. Labels are never read."""
    columns = resolve_entity_columns(events)
    src_col, dst_col = columns["src_ip"], columns["dst_ip"]
    meta: dict[str, Any] = {"available": False, "reason": None}
    if src_col is None or dst_col is None:
        meta["reason"] = "dataset has no Source/Destination IP columns"
        return nx.Graph(), meta

    work = events[[src_col, dst_col]].dropna().copy()
    if work.empty:
        meta["reason"] = "no records with both endpoints present"
        return nx.Graph(), meta

    work["__src"] = work[src_col].astype(str)
    work["__dst"] = work[dst_col].astype(str)
    aggregations: dict[str, Any] = {"n_flows": ("__dst", "size")}
    bytes_cols = [c for c in ("Total Length of Fwd Packets", "Total Length of Bwd Packets") if c in events.columns]
    if bytes_cols:
        work["__bytes"] = events.loc[work.index, bytes_cols].sum(axis=1)
        aggregations["bytes_total"] = ("__bytes", "sum")
    grouped = work.groupby(["__src", "__dst"], sort=False).agg(**aggregations)

    k = float(settings.graph.derived_confidence_k if k_confidence is None else k_confidence)
    graph = nx.Graph()
    for (source, target), row in grouped.iterrows():
        n_flows = int(row["n_flows"])
        confidence = float(1.0 - np.exp(-n_flows / max(k, 1e-9)))
        if graph.has_edge(source, target):
            data = graph[source][target]
            data["n_flows"] += n_flows
            data["confidence"] = float(1.0 - np.exp(-data["n_flows"] / max(k, 1e-9)))
        else:
            graph.add_edge(source, target, n_flows=n_flows, confidence=confidence,
                           bytes_total=float(row.get("bytes_total", 0.0) or 0.0))

    event_counts = pd.concat([work["__src"], work["__dst"]]).value_counts()
    for node in graph.nodes():
        graph.nodes[node]["is_internal"] = is_private_ip(node)
        graph.nodes[node]["n_events"] = int(event_counts.get(node, 0))

    meta.update(
        {
            "available": True,
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "n_components": nx.number_connected_components(graph),
            "largest_component": max((len(c) for c in nx.connected_components(graph)), default=0),
            "confidence_kernel": k,
        }
    )
    return graph, meta


def bfs_from_sources(graph: nx.Graph, sources: set[str], max_hops: int) -> dict[str, dict[str, Any]]:
    """BFS multi-source shortest distance, capped at ``max_hops``."""
    reachable: dict[str, dict[str, Any]] = {}
    if not sources or graph.number_of_nodes() == 0:
        return reachable
    queue: list[tuple[str, int, str]] = []
    for source in sources:
        if source in graph:
            reachable[source] = {"hops": 0, "path": [source], "via": source}
            queue.append((source, 0, source))
    while queue:
        node, distance, origin = queue.pop(0)
        if distance >= max_hops:
            continue
        for neighbor in graph.neighbors(node):
            if neighbor in reachable:
                continue
            reachable[neighbor] = {
                "hops": distance + 1,
                "path": reachable[node]["path"] + [neighbor],
                "via": origin,
            }
            queue.append((neighbor, distance + 1, origin))
    return reachable


def paths_to_intel(
    graph: nx.Graph, flagged_ips: set[str], max_hops: int
) -> dict[str, dict[str, Any]]:
    """Map every node within ``max_hops`` of a flagged IP to its shortest path."""
    present = {ip for ip in flagged_ips if ip in graph}
    if not present:
        return {}
    return bfs_from_sources(graph, present, max_hops)


def describe_path(
    graph: nx.Graph,
    path: list[str],
    intel_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn a hop list into structured, auditable evidence."""
    if not path:
        return {"available": False, "reason": "empty path"}
    hops: list[dict[str, Any]] = []
    for position, node in enumerate(path):
        entry: dict[str, Any] = {
            "position": position,
            "entity": node,
            "is_internal": is_private_ip(node),
        }
        if position < len(path) - 1:
            next_node = path[position + 1]
            if graph.has_edge(node, next_node):
                data = graph[node][next_node]
                entry["next_hop"] = next_node
                entry["observed_flows"] = int(data.get("n_flows", 0))
                entry["edge_confidence"] = round(float(data.get("confidence", 0.0)), 6)
                entry["confidence_rule"] = "aggregated: 1 - exp(-n_flows/K)"
        if position == len(path) - 1 and intel_index:
            info = (intel_index.get("ips") or {}).get(node)
            if info:
                entry["intel"] = {
                    "n_malicious_training_records": info.get("n_malicious_records"),
                    "n_total_training_records": info.get("n_total_records"),
                    "confidence": info.get("confidence"),
                    "source": intel_index.get("source"),
                }
        hops.append(entry)
    return {
        "available": True,
        "n_hops": len(path) - 1,
        "entities": path,
        "hops": hops,
        "interpretation": (
            "Shortest path over observed communications between the investigated entity and an "
            "IP with training-labelled malicious activity. Proximity is context, not evidence of "
            "malicious intent: benign hosts legitimately communicate with previously attacked hosts."
        ),
    }


def component_and_community_maps(graph: nx.Graph, settings: Settings) -> dict[str, Any]:
    """Component ids, component sizes and (optionally) deterministic communities."""
    result: dict[str, Any] = {"component_id": {}, "component_size": {}, "community_id": {}, "method": None}
    if graph.number_of_nodes() == 0:
        return result
    for index, component in enumerate(nx.connected_components(graph)):
        size = len(component)
        for node in component:
            result["component_id"][node] = index
            result["component_size"][node] = size

    if not settings.graph.community_detection:
        return result
    try:
        if hasattr(nx.community, "louvain_communities"):
            communities = nx.community.louvain_communities(graph, seed=settings.seed)
            result["method"] = "louvain_communities(seed=config.seed)"
        else:  # pragma: no cover - depends on networkx version
            communities = nx.community.greedy_modularity_communities(graph)
            result["method"] = "greedy_modularity_communities"
        for index, community in enumerate(communities):
            for node in community:
                result["community_id"][node] = index
        result["n_communities"] = len(communities)
    except Exception as exc:  # community detection is optional context
        result["method"] = f"unavailable ({type(exc).__name__}: {exc})"
        logger.warning("community detection failed: %s", exc)
    return result
