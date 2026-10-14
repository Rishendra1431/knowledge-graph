"""Graph-derived features for the graph-aware detector.

Every feature is either

* **structural** - computed from the *observed* entity graph. Labels never enter
  this computation. In ``graph.structural_scope = full`` the graph covers all
  evaluated events (transductive setting); in ``train_only`` it is built from
  training events only and unseen entities fall back to documented sentinels
  (inductive setting). Both configurations are reported.
* **relational-supervised** - computed from the *training labels only* through the
  threat-intel index (``src/graph/builder.py::build_intel_index``). These are
  legitimate supervised features: no test label ever contributes.

Each feature has an entry in :data:`FEATURE_DESCRIPTIONS` stating what it means,
where it comes from and whether supervised information is involved. That
metadata is written into the results artifact and shown in the dashboard, so an
explanation can never claim more than the feature actually encodes.
"""
from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.config import Settings, get_logger
from src.data.preprocess import is_private_ip, port_to_service
from src.graph.builder import intel_confidence
from src.graph.paths import (
    build_ip_graph,
    component_and_community_maps,
    paths_to_intel,
    resolve_entity_columns,
)

logger = get_logger("graph.features")

# name -> (description, provenance, uses_training_labels)
class FeatureDescription(tuple):
    """Tuple-compatible feature metadata with readable field access."""

    __slots__ = ()

    def __new__(cls, values: tuple[str, str, bool]) -> "FeatureDescription":
        return super().__new__(cls, values)

    def __getitem__(self, key: int | str) -> Any:
        if isinstance(key, str):
            return super().__getitem__({
                "description": 0,
                "source": 1,
                "provenance": 1,
                "uses_training_labels": 2,
            }[key])
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


_FEATURE_DESCRIPTION_VALUES: dict[str, tuple[str, str, bool]] = {
    "g_src_degree": ("Distinct IPs the source address communicated with in the graph scope", "structural_ip_graph", False),
    "g_dst_degree": ("Distinct IPs the destination address communicated with in the graph scope", "structural_ip_graph", False),
    "g_src_weighted_degree": ("Total observed flows involving the source address", "structural_ip_graph", False),
    "g_dst_weighted_degree": ("Total observed flows involving the destination address", "structural_ip_graph", False),
    "g_src_fanout": ("Distinct destinations contacted by the source address (outgoing breadth)", "event_endpoints", False),
    "g_dst_fanin": ("Distinct sources contacting the destination address (incoming breadth)", "event_endpoints", False),
    "g_src_event_count": ("Records involving the source address in the graph scope", "event_endpoints", False),
    "g_dst_event_count": ("Records involving the destination address in the graph scope", "event_endpoints", False),
    "g_src_event_share": ("Share of all in-scope records that involve the source address", "event_endpoints", False),
    "g_dst_event_share": ("Share of all in-scope records that involve the destination address", "event_endpoints", False),
    "g_degree_ratio": ("(src_degree + 1) / (dst_degree + 1); large values indicate one-sided fan-out", "structural_ip_graph", False),
    "g_common_neighbors": ("Shared neighbours of source and destination in the IP graph", "structural_ip_graph", False),
    "g_jaccard_neighbors": ("Jaccard overlap of the two neighbour sets", "structural_ip_graph", False),
    "g_src_clustering": ("Local clustering coefficient of the source address", "structural_ip_graph", False),
    "g_dst_clustering": ("Local clustering coefficient of the destination address", "structural_ip_graph", False),
    "g_src_component_size": ("Size of the connected component containing the source address", "structural_ip_graph", False),
    "g_dst_component_size": ("Size of the connected component containing the destination address", "structural_ip_graph", False),
    "g_same_component": ("1 if both endpoints sit in the same connected component", "structural_ip_graph", False),
    "g_src_community": ("Community id of the source address (deterministic seeded partition)", "structural_ip_graph", False),
    "g_dst_community": ("Community id of the destination address", "structural_ip_graph", False),
    "g_same_community": ("1 if both endpoints were placed in the same community", "structural_ip_graph", False),
    "g_src_is_internal": ("1 if the source address is an RFC1918/loopback address", "event_endpoints", False),
    "g_dst_is_internal": ("1 if the destination address is an RFC1918/loopback address", "event_endpoints", False),
    "g_src_unique_dst_ports": ("Distinct destination ports the source address touched", "event_endpoints", False),
    "g_dst_unique_dst_ports": ("Distinct destination ports observed on the destination address", "event_endpoints", False),
    "g_src_unique_services": ("Distinct services (port + well-known name) the source address used", "event_endpoints", False),
    "g_dst_unique_services": ("Distinct services observed on the destination address", "event_endpoints", False),
    "g_src_protocol_diversity": ("Distinct protocols used by the source address", "event_endpoints", False),
    "g_src_intel_malicious_count": ("Training-labelled malicious records involving the source address", "intel_index(train labels)", True),
    "g_dst_intel_malicious_count": ("Training-labelled malicious records involving the destination address", "intel_index(train labels)", True),
    "g_src_intel_ratio": ("Laplace-smoothed malicious ratio for the source address: (n_mal + 1)/(n_tot + 2)", "intel_index(train labels)", True),
    "g_dst_intel_ratio": ("Laplace-smoothed malicious ratio for the destination address", "intel_index(train labels)", True),
    "g_src_is_flagged_intel": ("1 if the source address appears in >=1 training-labelled malicious record", "intel_index(train labels)", True),
    "g_dst_is_flagged_intel": ("1 if the destination address appears in >=1 training-labelled malicious record", "intel_index(train labels)", True),
    "g_src_neighbors_flagged": ("Number of direct neighbours of the source address carrying intel evidence", "intel_index(train labels)", True),
    "g_dst_neighbors_flagged": ("Number of direct neighbours of the destination address carrying intel evidence", "intel_index(train labels)", True),
    "g_src_hops_to_intel": ("Graph distance from the source address to the nearest intel-flagged IP (NaN if > max_hops)", "intel_index(train labels)", True),
    "g_dst_hops_to_intel": ("Graph distance from the destination address to the nearest intel-flagged IP", "intel_index(train labels)", True),
}

FEATURE_DESCRIPTIONS: dict[str, FeatureDescription] = {
    name: FeatureDescription(values) for name, values in _FEATURE_DESCRIPTION_VALUES.items()
}


class GraphFeatureExtractor:
    """Fit on the structural scope + training labels; transform any event frame."""

    def __init__(self, settings: Settings, structural_scope: str | None = None) -> None:
        self.settings = settings
        self.scope = structural_scope or settings.graph.structural_scope
        self.ip_graph: nx.Graph = nx.Graph()
        self.maps: dict[str, Any] = {}
        self.intel_index: dict[str, Any] = {}
        self.intel_paths: dict[str, dict[str, Any]] = {}
        self.feature_names_: list[str] = []
        self.fit_report: dict[str, Any] = {}
        self._per_ip: dict[str, pd.Series] = {}
        self._n_scope_events: int = 0
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, scope_events: pd.DataFrame, intel_index: dict[str, Any]) -> "GraphFeatureExtractor":
        """Build the structural graph and the per-IP statistics.

        ``scope_events`` is the frame that defines the structural scope. It must
        never be selected using the test labels.
        """
        self.intel_index = intel_index or {"available": False, "ips": {}}
        columns = resolve_entity_columns(scope_events)
        self._columns = columns
        src_col, dst_col = columns["src_ip"], columns["dst_ip"]
        dst_port_col = columns["dst_port"]
        proto_col = columns["protocol"]
        self._n_scope_events = int(len(scope_events))

        if src_col is None or dst_col is None:
            self.feature_names_ = []
            self.fit_report = {
                "structural_scope": self.scope,
                "available": False,
                "reason": "dataset provides no Source/Destination IP columns; graph features disabled",
            }
            self._fitted = True
            logger.warning("graph features disabled: no IP columns")
            return self

        self.ip_graph, graph_meta = build_ip_graph(scope_events, self.settings)
        self.maps = component_and_community_maps(self.ip_graph, self.settings)

        work = scope_events[[src_col, dst_col]].dropna().copy()
        work["__src"] = work[src_col].astype(str)
        work["__dst"] = work[dst_col].astype(str)

        # In- and out-degree computed on the *directed* event view, which is what
        # distinguishes a scanner (one source, many destinations) from a server.
        fanout = work.groupby("__src")["__dst"].nunique()
        fanin = work.groupby("__dst")["__src"].nunique()
        src_events = work["__src"].value_counts()
        dst_events = work["__dst"].value_counts()
        src_is_internal = pd.Series(
            work["__src"].drop_duplicates().map(is_private_ip).to_numpy(),
            index=pd.Index(work["__src"].drop_duplicates(), name="__src"),
            dtype="float64",
        )
        dst_is_internal = pd.Series(
            work["__dst"].drop_duplicates().map(is_private_ip).to_numpy(),
            index=pd.Index(work["__dst"].drop_duplicates(), name="__dst"),
            dtype="float64",
        )

        port_diversity_src = port_diversity_dst = services_src = services_dst = protocol_div = None
        if dst_port_col is not None:
            services = scope_events[dst_port_col].map(port_to_service)
            pairs = pd.DataFrame(
                {
                    "__src": scope_events[src_col].astype(str),
                    "__dst": scope_events[dst_col].astype(str),
                    "__port": scope_events[dst_port_col],
                    "__svc": services,
                }
            ).dropna(subset=["__port"])
            port_diversity_src = pairs.groupby("__src")["__port"].nunique()
            port_diversity_dst = pairs.groupby("__dst")["__port"].nunique()
            services_src = pairs.groupby("__src")["__svc"].nunique()
            services_dst = pairs.groupby("__dst")["__svc"].nunique()
        if proto_col is not None:
            proto_pairs = pd.DataFrame(
                {
                    "__src": scope_events[src_col].astype(str),
                    "__dst": scope_events[dst_col].astype(str),
                    "__proto": scope_events[proto_col],
                }
            ).dropna(subset=["__proto"])
            protocol_div = proto_pairs.groupby("__src")["__proto"].nunique()
            protocol_diversity_dst = proto_pairs.groupby("__dst")["__proto"].nunique()
        else:
            protocol_diversity_dst = None

        degrees = dict(self.ip_graph.degree())
        weighted = dict(self.ip_graph.degree(weight="n_flows"))
        clustering = nx.clustering(self.ip_graph, weight="n_flows") if self.ip_graph.number_of_nodes() else {}

        self._per_ip = {
            "degree": pd.Series(degrees, dtype="float64"),
            "weighted_degree": pd.Series(weighted, dtype="float64"),
            "clustering": pd.Series(clustering, dtype="float64"),
            "fanout": fanout.astype("float64"),
            "fanin": fanin.astype("float64"),
            "src_events": src_events.astype("float64"),
            "dst_events": dst_events.astype("float64"),
            "is_internal": src_is_internal.combine_first(dst_is_internal).astype("float64") if src_is_internal is not None else pd.Series(dtype="float64"),
            "port_div_src": port_diversity_src.astype("float64") if port_diversity_src is not None else None,
            "port_div_dst": port_diversity_dst.astype("float64") if port_diversity_dst is not None else None,
            "services_src": services_src.astype("float64") if services_src is not None else None,
            "services_dst": services_dst.astype("float64") if services_dst is not None else None,
            "proto_div_src": protocol_div.astype("float64") if protocol_div is not None else None,
        }
        self._protocol_div_dst = protocol_diversity_dst.astype("float64") if protocol_diversity_dst is not None else None

        intel_ips = self.intel_index.get("ips", {}) if self.intel_index.get("available") else {}
        self._intel_ratio = pd.Series(
            {ip: float(info.get("confidence", intel_confidence(0, 0))) for ip, info in intel_ips.items()},
            dtype="float64",
        )
        self._intel_count = pd.Series(
            {ip: float(info.get("n_malicious_records", 0)) for ip, info in intel_ips.items()},
            dtype="float64",
        )
        flagged = set(self.intel_index.get("flagged_ips", [])) if self.intel_index.get("available") else set()
        self._flagged = flagged
        self.intel_paths = paths_to_intel(self.ip_graph, flagged, self.settings.graph.max_hops_to_intel)

        neighbor_flagged: dict[str, int] = {}
        for node in self.ip_graph.nodes():
            neighbor_flagged[node] = sum(1 for n in self.ip_graph.neighbors(node) if n in flagged)
        self._neighbors_flagged = pd.Series(neighbor_flagged, dtype="float64")

        self.feature_names_ = list(FEATURE_DESCRIPTIONS.keys())
        self.fit_report = {
            "structural_scope": self.scope,
            "available": True,
            "n_scope_events": self._n_scope_events,
            "ip_graph": graph_meta,
            "community_detection": self.maps.get("method"),
            "n_communities": self.maps.get("n_communities"),
            "intel_available": bool(self.intel_index.get("available")),
            "n_flagged_ips": len(flagged),
            "n_ips_with_path_to_intel": len(self.intel_paths),
            "feature_names": self.feature_names_,
            "n_features": len(self.feature_names_),
            "supervised_features": [k for k, v in FEATURE_DESCRIPTIONS.items() if v["uses_training_labels"]],
            "sentinel_policy": (
                "Structural lookups for entities absent from the structure scope return 0.0 "
                "(unknown entity), graph distances return NaN (beyond max_hops) and intel ratios "
                "return the Laplace prior 0.5. The count of unseen entities per call is recorded "
                "in 'unseen_entity_fraction'."
            ),
            "scope_note": (
                "full = transductive graph over all evaluated events (no labels involved); "
                "train_only = inductive graph from training events only."
            ),
        }
        self._fitted = True
        logger.info(
            "graph features fitted (scope=%s): %d features, IP graph %d nodes / %d edges, %d IPs with a path to intel",
            self.scope, len(self.feature_names_), self.ip_graph.number_of_nodes(),
            self.ip_graph.number_of_edges(), len(self.intel_paths),
        )
        return self

    # ------------------------------------------------------------------
    def _lookup(self, series: pd.Series, mapping: pd.Series, default: float) -> np.ndarray:
        if mapping is None or mapping.empty:
            return np.full(len(series), default, dtype="float64")
        return series.map(mapping).astype("float64").fillna(default).to_numpy()

    def transform(self, events: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("GraphFeatureExtractor.transform called before fit")
        if not self.feature_names_:
            return pd.DataFrame(index=events.index)

        columns = resolve_entity_columns(events)
        src_col, dst_col = columns["src_ip"], columns["dst_ip"]
        src = events[src_col].astype(str) if src_col else pd.Series(["unknown"] * len(events), index=events.index)
        dst = events[dst_col].astype(str) if dst_col else pd.Series(["unknown"] * len(events), index=events.index)

        out = pd.DataFrame(index=events.index)
        out["g_src_degree"] = self._lookup(src, self._per_ip["degree"], 0.0)
        out["g_dst_degree"] = self._lookup(dst, self._per_ip["degree"], 0.0)
        out["g_src_weighted_degree"] = self._lookup(src, self._per_ip["weighted_degree"], 0.0)
        out["g_dst_weighted_degree"] = self._lookup(dst, self._per_ip["weighted_degree"], 0.0)
        out["g_src_fanout"] = self._lookup(src, self._per_ip["fanout"], 0.0)
        out["g_dst_fanin"] = self._lookup(dst, self._per_ip["fanin"], 0.0)
        out["g_src_event_count"] = self._lookup(src, self._per_ip["src_events"], 0.0)
        out["g_dst_event_count"] = self._lookup(dst, self._per_ip["dst_events"], 0.0)
        denominator = max(self._n_scope_events, 1)
        out["g_src_event_share"] = out["g_src_event_count"] / denominator
        out["g_dst_event_share"] = out["g_dst_event_count"] / denominator
        out["g_degree_ratio"] = (out["g_src_degree"] + 1.0) / (out["g_dst_degree"] + 1.0)

        common, jaccard = self._pair_features(src, dst)
        out["g_common_neighbors"] = common
        out["g_jaccard_neighbors"] = jaccard

        out["g_src_clustering"] = self._lookup(src, self._per_ip["clustering"], 0.0)
        out["g_dst_clustering"] = self._lookup(dst, self._per_ip["clustering"], 0.0)
        component_size = pd.Series(self.maps.get("component_size", {}), dtype="float64")
        component_id = pd.Series(self.maps.get("component_id", {}), dtype="float64")
        community_id = pd.Series(self.maps.get("community_id", {}), dtype="float64")
        out["g_src_component_size"] = self._lookup(src, component_size, 0.0)
        out["g_dst_component_size"] = self._lookup(dst, component_size, 0.0)
        src_component = src.map(component_id) if not component_id.empty else pd.Series([np.nan] * len(events), index=events.index)
        dst_component = dst.map(component_id) if not component_id.empty else pd.Series([np.nan] * len(events), index=events.index)
        out["g_same_component"] = (src_component == dst_component).astype("float64")
        out["g_src_community"] = self._lookup(src, community_id, -1.0)
        out["g_dst_community"] = self._lookup(dst, community_id, -1.0)
        src_community = src.map(community_id) if not community_id.empty else pd.Series([np.nan] * len(events), index=events.index)
        dst_community = dst.map(community_id) if not community_id.empty else pd.Series([np.nan] * len(events), index=events.index)
        out["g_same_community"] = ((src_community == dst_community) & src_community.notna()).astype("float64")

        out["g_src_is_internal"] = src.map(is_private_ip).astype("float64")
        out["g_dst_is_internal"] = dst.map(is_private_ip).astype("float64")
        out["g_src_unique_dst_ports"] = self._lookup(src, self._per_ip["port_div_src"], 0.0)
        out["g_dst_unique_dst_ports"] = self._lookup(dst, self._per_ip["port_div_dst"], 0.0)
        out["g_src_unique_services"] = self._lookup(src, self._per_ip["services_src"], 0.0)
        out["g_dst_unique_services"] = self._lookup(dst, self._per_ip["services_dst"], 0.0)
        out["g_src_protocol_diversity"] = self._lookup(src, self._per_ip["proto_div_src"], 0.0)

        out["g_src_intel_malicious_count"] = self._lookup(src, self._intel_count, 0.0)
        out["g_dst_intel_malicious_count"] = self._lookup(dst, self._intel_count, 0.0)
        out["g_src_intel_ratio"] = self._lookup(src, self._intel_ratio, intel_confidence(0, 0))
        out["g_dst_intel_ratio"] = self._lookup(dst, self._intel_ratio, intel_confidence(0, 0))
        out["g_src_is_flagged_intel"] = src.isin(self._flagged).astype("float64")
        out["g_dst_is_flagged_intel"] = dst.isin(self._flagged).astype("float64")
        out["g_src_neighbors_flagged"] = self._lookup(src, self._neighbors_flagged, 0.0)
        out["g_dst_neighbors_flagged"] = self._lookup(dst, self._neighbors_flagged, 0.0)
        out["g_src_hops_to_intel"] = self._hop_features(src)
        out["g_dst_hops_to_intel"] = self._hop_features(dst)

        # Entity coverage: in the inductive scope many test entities are unseen.
        known = set(self.ip_graph.nodes())
        unseen = (~src.isin(known)) | (~dst.isin(known))
        self._last_unseen_fraction = float(unseen.mean()) if len(events) else 0.0

        return out[self.feature_names_]

    # ------------------------------------------------------------------
    def _pair_features(self, src: pd.Series, dst: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        """Common neighbours and Jaccard overlap, computed once per unique pair."""
        cap = int(self.settings.graph.max_common_neighbor_pairs)
        pairs = pd.DataFrame({"src": src.to_numpy(), "dst": dst.to_numpy()}).drop_duplicates()
        truncated = False
        if len(pairs) > cap:
            truncated = True
            pairs = pairs.head(cap)

        neighbors = {node: set(self.ip_graph.neighbors(node)) for node in self.ip_graph.nodes()}
        values_common = np.zeros(len(pairs), dtype="float64")
        values_jaccard = np.zeros(len(pairs), dtype="float64")
        for position, (left, right) in enumerate(zip(pairs["src"].to_numpy(), pairs["dst"].to_numpy())):
            left_set = neighbors.get(left)
            right_set = neighbors.get(right)
            if left_set is None or right_set is None:
                continue
            intersection = len(left_set & right_set)
            union = len(left_set | right_set)
            values_common[position] = float(intersection)
            values_jaccard[position] = float(intersection / union) if union else 0.0

        index = pd.MultiIndex.from_arrays([pairs["src"].to_numpy(), pairs["dst"].to_numpy()])
        lookup_common = pd.Series(values_common, index=index)
        lookup_jaccard = pd.Series(values_jaccard, index=index)
        event_index = pd.MultiIndex.from_arrays([src.to_numpy(), dst.to_numpy()])
        common = lookup_common.reindex(event_index).fillna(0.0).to_numpy()
        jaccard = lookup_jaccard.reindex(event_index).fillna(0.0).to_numpy()
        self._last_pair_truncated = truncated
        return common, jaccard

    def _hop_features(self, addresses: pd.Series) -> np.ndarray:
        if not self.intel_paths:
            return np.full(len(addresses), np.nan)
        hops = {ip: entry["hops"] for ip, entry in self.intel_paths.items() if entry["hops"] > 0}
        if not hops:
            return np.full(len(addresses), np.nan)
        return addresses.map(pd.Series(hops, dtype="float64")).to_numpy()

    # ------------------------------------------------------------------
    def path_evidence(self, address: str, max_hops: int | None = None) -> dict[str, Any]:
        """Structured path evidence for one address (used by the explainer)."""
        from src.graph.paths import describe_path

        entry = self.intel_paths.get(address)
        if entry is None:
            return {
                "available": False,
                "reason": (
                    "no observed path within "
                    f"{max_hops or self.settings.graph.max_hops_to_intel} hops to an IP with "
                    "training-labelled malicious activity"
                ),
            }
        if entry["hops"] == 0:
            return {
                "available": True,
                "n_hops": 0,
                "entities": [address],
                "hops": [{"position": 0, "entity": address, "is_internal": is_private_ip(address),
                          "intel": (self.intel_index.get("ips") or {}).get(address)}],
                "interpretation": "The address itself carries training-derived threat-intel evidence; this is context, not proof of malicious intent.",
            }
        return describe_path(self.ip_graph, entry["path"], self.intel_index)

    def report(self) -> dict[str, Any]:
        payload = dict(self.fit_report)
        payload["last_unseen_entity_fraction"] = getattr(self, "_last_unseen_fraction", None)
        payload["last_pair_computation_truncated"] = getattr(self, "_last_pair_truncated", None)
        payload["feature_documentation"] = FEATURE_DESCRIPTIONS
        return payload
