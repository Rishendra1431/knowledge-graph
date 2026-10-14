"""Knowledge graph construction over observed security events.

Entity and relationship model
-----------------------------
Nodes (only created when the dataset actually supports them):

===================  ==============  ==================================================
node type            pattern         created from
===================  ==============  ==================================================
ip                   ``ip:<addr>``   Source IP / Destination IP column
host                 ``host:<addr>`` internal (RFC1918) addresses, one host per address
service              ``svc:<p>/<n>`` Destination Port + protocol (well-known name lookup)
protocol             ``proto:<n>``   Protocol column
event                ``evt:<id>``    one node per flow record (SecurityEvent)
technique            ``tech:<name>`` dataset label mapped through our documented taxonomy
intel                ``intel:<addr>``IPs observed in *training-labelled malicious* events
===================  ==============  ==================================================

Relationships: ``COMMUNICATES_WITH`` (aggregated host pair), ``USES_SERVICE``,
``RUNS_ON_PROTOCOL``, ``INITIATED_BY``, ``TARGETED``, ``MATCHES_TECHNIQUE``
(labels required), ``FLAGGED_AS_INTEL``.

Confidence is never invented per edge. Three documented rules are used, each
recorded on the edge in ``confidence_rule``:

* ``observed``: 1.0 - the dataset row states the fact directly.
* ``aggregated``: ``1 - exp(-evidence_count / K)`` with ``K = graph.derived_confidence_k``.
  Monotone in the number of supporting events, saturating at 1.
* ``intel_smoothed_ratio``: ``(n_malicious + 1) / (n_total + 2)`` (Laplace smoothing),
  where ``n_malicious`` counts **training-labelled** malicious events only.

Everything is traceable: each node/edge carries a ``provenance_id`` that resolves
to a dataset file list, a row-id sample and the code path that created it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import networkx as nx
import numpy as np
import pandas as pd

from src.config import Settings, get_logger
from src.data.preprocess import CATEGORY_COLUMN, DAY_INTERNAL, TIMESTAMP_INTERNAL, is_private_ip, port_to_service

logger = get_logger("graph.builder")

OBSERVED_CONFIDENCE = 1.0
PROVENANCE_SAMPLE_SIZE = 20

NODE_TYPES = ("ip", "host", "service", "protocol", "event", "technique", "intel")
RELATION_TYPES = (
    "COMMUNICATES_WITH",
    "USES_SERVICE",
    "RUNS_ON_PROTOCOL",
    "INITIATED_BY",
    "TARGETED",
    "MATCHES_TECHNIQUE",
    "FLAGGED_AS_INTEL",
    "ALIAS_OF",
)


@dataclass
class Provenance:
    provenance_id: str
    dataset: str
    origin: str
    created_by: str
    files: list[str] = field(default_factory=list)
    record_id_sample: list[int] = field(default_factory=list)
    n_supporting_records: int = 0
    created_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance_id": self.provenance_id,
            "dataset": self.dataset,
            "origin": self.origin,
            "created_by": self.created_by,
            "files": self.files,
            "record_id_sample": self.record_id_sample,
            "n_supporting_records": self.n_supporting_records,
            "created_at_utc": self.created_at_utc,
        }


def aggregated_confidence(evidence_count: int, k: float) -> float:
    """1 - exp(-n/K); K is configurable and documented in docs/ARCHITECTURE.md."""
    if evidence_count <= 0:
        return 0.0
    return float(1.0 - math.exp(-evidence_count / max(k, 1e-9)))


def intel_confidence(n_malicious: int, n_total: int) -> float:
    """Laplace-smoothed malicious ratio. Zero evidence -> 0.5, never 0 or 1."""
    return float((n_malicious + 1) / (n_total + 2))


def _first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


class KnowledgeGraph:
    """Wrapper around networkx holding the graph plus a provenance registry."""

    def __init__(self, name: str, settings: Settings, dataset_origin: str = "real_csv") -> None:
        self.name = name
        self.settings = settings
        self.graph = nx.MultiDiGraph(name=name)
        self.provenance: dict[str, Provenance] = {}
        self.dataset_origin = dataset_origin
        self._provenance_counter = 0
        self.build_log: list[str] = []

    # -- provenance --------------------------------------------------------
    def add_provenance(
        self,
        created_by: str,
        files: list[str] | None = None,
        record_sample: list[int] | None = None,
        n_supporting_records: int = 0,
    ) -> str:
        self._provenance_counter += 1
        pid = f"prov_{self._provenance_counter:06d}"
        self.provenance[pid] = Provenance(
            provenance_id=pid,
            dataset=self.name,
            origin=self.dataset_origin,
            created_by=created_by,
            files=sorted(set(files or []))[:50],
            record_id_sample=[int(r) for r in (record_sample or [])][:PROVENANCE_SAMPLE_SIZE],
            n_supporting_records=int(n_supporting_records),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        return pid

    def provenance_of(self, provenance_id: str) -> dict[str, Any] | None:
        entry = self.provenance.get(provenance_id)
        return entry.to_dict() if entry else None

    # -- mutation helpers --------------------------------------------------
    def add_node(self, node_id: str, node_type: str, **attributes: Any) -> None:
        if node_id in self.graph:
            existing = self.graph.nodes[node_id]
            existing["evidence_count"] = int(existing.get("evidence_count", 0)) + int(
                attributes.get("evidence_count", 1)
            )
            for key in ("last_seen", "label"):
                if attributes.get(key) is not None:
                    existing[key] = attributes[key]
            return
        payload = {"node_type": node_type, "label": str(node_id), "evidence_count": 1}
        payload.update({k: v for k, v in attributes.items() if v is not None})
        self.graph.add_node(node_id, **payload)

    def add_edge(self, source: str, target: str, relation: str, **attributes: Any) -> None:
        if source not in self.graph or target not in self.graph:
            raise KeyError(f"both endpoints must exist before adding an edge ({source} -> {target})")
        key = self.graph.add_edge(source, target, relation=relation, **attributes)
        self.graph.edges[source, target, key]["relation"] = relation

    # -- summaries ---------------------------------------------------------
    def node_type_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            node_type = data.get("node_type", "unknown")
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts

    def relation_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            relation = data.get("relation", "unknown")
            counts[relation] = counts.get(relation, 0) + 1
        return counts

    def confidence_stats(self) -> dict[str, Any]:
        values = [float(d.get("confidence", 0.0)) for _, _, d in self.graph.edges(data=True)]
        if not values:
            return {"n_edges": 0, "min": None, "max": None, "mean": None, "median": None}
        array = np.asarray(values)
        return {
            "n_edges": int(len(values)),
            "min": round(float(array.min()), 6),
            "max": round(float(array.max()), 6),
            "mean": round(float(array.mean()), 6),
            "median": round(float(np.median(array)), 6),
            "n_edges_confidence_1": int((array >= OBSERVED_CONFIDENCE).sum()),
        }

    def summary(self) -> dict[str, Any]:
        undirected = self.graph.to_undirected()
        components = list(nx.connected_components(undirected)) if undirected.number_of_nodes() else []
        largest = max((len(c) for c in components), default=0)
        return {
            "name": self.name,
            "n_nodes": int(self.graph.number_of_nodes()),
            "n_edges": int(self.graph.number_of_edges()),
            "node_types": self.node_type_counts(),
            "relations": self.relation_counts(),
            "confidence": self.confidence_stats(),
            "n_connected_components": len(components),
            "largest_component_nodes": int(largest),
            "n_provenance_records": len(self.provenance),
            "is_isolate_free_ratio": self._isolate_free_ratio(),
            "build_log": self.build_log,
        }

    def _isolate_free_ratio(self) -> float | None:
        if self.graph.number_of_nodes() == 0:
            return None
        isolates = len(list(nx.isolates(self.graph)))
        return round(1.0 - isolates / self.graph.number_of_nodes(), 6)

    # -- export ------------------------------------------------------------
    def to_node_link(self, max_nodes: int | None = None, node_ids: list[str] | None = None) -> dict[str, Any]:
        """Compact JSON for the dashboard. Attributes are flattened to simple types."""
        if node_ids is not None:
            selected = [n for n in node_ids if n in self.graph][:max_nodes] if max_nodes else [n for n in node_ids if n in self.graph]
        elif max_nodes is not None and self.graph.number_of_nodes() > max_nodes:
            degree = dict(self.graph.degree())
            selected = sorted(degree, key=lambda n: degree[n], reverse=True)[:max_nodes]
        else:
            selected = list(self.graph.nodes())
        keep = set(selected)
        nodes = [
            {
                "id": node_id,
                "label": str(data.get("label", node_id)),
                "node_type": data.get("node_type", "unknown"),
                "confidence": data.get("confidence"),
                "evidence_count": data.get("evidence_count"),
                "is_internal": data.get("is_internal"),
            }
            for node_id, data in self.graph.nodes(data=True)
            if node_id in keep
        ]
        edges = [
            {
                "source": source,
                "target": target,
                "relation": data.get("relation"),
                "confidence": data.get("confidence"),
                "evidence_count": data.get("evidence_count"),
                "confidence_rule": data.get("confidence_rule"),
            }
            for source, target, data in self.graph.edges(data=True)
            if source in keep and target in keep
        ]
        return {"nodes": nodes, "edges": edges, "n_nodes_total": self.graph.number_of_nodes(),
                "n_edges_total": self.graph.number_of_edges(), "truncated": len(nodes) < self.graph.number_of_nodes()}

    def ego_graph_edges(self, center: str, radius: int = 1) -> tuple[list[str], list[dict[str, Any]]]:
        if center not in self.graph:
            return [], []
        undirected = self.graph.to_undirected()
        nodes = list(nx.ego_graph(undirected, center, radius=radius).nodes())
        keep = set(nodes)
        edges = [
            {"source": source, "target": target, "relation": data.get("relation"),
             "confidence": data.get("confidence")}
            for source, target, data in self.graph.edges(data=True)
            if source in keep and target in keep
        ]
        return nodes, edges

    def to_graphml(self, path: str) -> str:
        """GraphML export. Attribute values are coerced to GraphML-safe types."""
        simple = nx.MultiDiGraph()
        for node_id, data in self.graph.nodes(data=True):
            simple.add_node(node_id, **{k: _safe(v) for k, v in data.items()})
        for source, target, data in self.graph.edges(data=True):
            simple.add_edge(source, target, **{k: _safe(v) for k, v in data.items()})
        nx.write_graphml(simple, path)
        return path


def _safe(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return ""
        return value
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(v) for v in value)
    return str(value)


# ---------------------------------------------------------------------------
# Threat-intel index from training labels
# ---------------------------------------------------------------------------
def build_intel_index(train_df: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    """Map each IP to its *training-labelled* malicious evidence.

    This is a supervised, hindsight-derived index. In deployment it would come
    from an external threat-intelligence feed; here it is derived strictly from
    the training labels, which is why it can be applied to the test split
    without leakage. Every explanation that uses it says so explicitly.
    """
    target = settings.preprocess.target_column
    src_col = _first_existing(train_df, ["Source IP"])
    dst_col = _first_existing(train_df, ["Destination IP"])
    if src_col is None or dst_col is None:
        logger.warning("no IP columns available; threat-intel index cannot be built")
        return {"available": False, "reason": "dataset has no Source/Destination IP columns", "ips": {}}

    stats: dict[str, dict[str, int]] = {}

    def _accumulate(frame: pd.DataFrame, ip_col: str) -> None:
        grouped = frame.groupby(ip_col, dropna=True)[target].agg(["size", "sum"])
        for ip, row in grouped.iterrows():
            entry = stats.setdefault(str(ip), {"n_total": 0, "n_malicious": 0})
            entry["n_total"] += int(row["size"])
            entry["n_malicious"] += int(row["sum"])

    attack_rows = train_df.loc[train_df[target] == 1]
    benign_rows = train_df.loc[train_df[target] == 0]
    _accumulate(train_df, src_col)
    _accumulate(train_df, dst_col)
    # A second pass over malicious rows is redundant with the above but makes the
    # source of n_malicious explicit and robust to future refactors.
    malicious_counts: dict[str, int] = {}
    for frame, column in ((attack_rows, src_col), (attack_rows, dst_col)):
        for ip, count in frame[column].value_counts().items():
            malicious_counts[str(ip)] = malicious_counts.get(str(ip), 0) + int(count)
    for ip, count in malicious_counts.items():
        stats.setdefault(ip, {"n_total": int(count), "n_malicious": 0})["n_malicious"] = count

    ips = {
        ip: {
            "n_total_records": entry["n_total"],
            "n_malicious_records": entry["n_malicious"],
            "confidence": round(intel_confidence(entry["n_malicious"], entry["n_total"]), 6),
            "is_internal": is_private_ip(ip),
        }
        for ip, entry in stats.items()
    }
    flagged = {ip for ip, entry in ips.items() if entry["n_malicious_records"] > 0}
    logger.info(
        "threat-intel index: %d IPs observed in training, %d with training-labelled malicious activity",
        len(ips), len(flagged),
    )
    return {
        "available": True,
        "source": "training labels only (derived threat-intel index, not an external feed)",
        "n_ips": len(ips),
        "n_flagged_ips": len(flagged),
        "flagged_ips": sorted(flagged),
        "ips": ips,
        "n_training_benign_records": int(len(benign_rows)),
        "n_training_malicious_records": int(len(attack_rows)),
    }


# ---------------------------------------------------------------------------
# Building the graph
# ---------------------------------------------------------------------------
def build_knowledge_graph(
    events: pd.DataFrame,
    settings: Settings,
    name: str = "cyber-threat-kg",
    include_labels: bool = False,
    intel_index: dict[str, Any] | None = None,
    dataset_origin: str = "real_csv",
) -> KnowledgeGraph:
    """Build the graph from observed events only.

    ``include_labels`` controls whether ``MATCHES_TECHNIQUE`` edges are added.
    They are added for the training/analysis graph (so a human can inspect which
    technique a flow was labelled with) and for alert explanations; they are
    never used as model input.
    """
    if events.empty:
        raise ValueError("cannot build a knowledge graph from an empty event frame")

    settings_entity = settings.preprocess.entity_columns
    src_col = _first_existing(events, [c for c in settings_entity if "Source IP" in c] or ["Source IP"])
    dst_col = _first_existing(events, [c for c in settings_entity if "Destination IP" in c] or ["Destination IP"])
    sport_col = _first_existing(events, [c for c in settings_entity if "Source Port" in c] or ["Source Port"])
    dport_col = _first_existing(events, [c for c in settings_entity if "Destination Port" in c] or ["Destination Port"])
    proto_col = _first_existing(events, [c for c in settings_entity if "Protocol" in c] or ["Protocol"])
    record_col = "__record_id" if "__record_id" in events.columns else None
    time_col = TIMESTAMP_INTERNAL if TIMESTAMP_INTERNAL in events.columns else None
    file_col = "__source_file" if "__source_file" in events.columns else None

    kg = KnowledgeGraph(name=name, settings=settings, dataset_origin=dataset_origin)
    has_timestamps = time_col is not None and events[time_col].notna().any()
    has_ips = src_col is not None and dst_col is not None

    if not has_ips:
        kg.build_log.append(
            "Dataset provides no Source/Destination IP columns: the graph is limited to "
            "event/service/protocol entities and their relationships."
        )
        logger.warning("building a reduced graph (no IP entities available)")

    def _times(frame: pd.DataFrame) -> tuple[str | None, str | None]:
        if not has_timestamps:
            return None, None
        values = frame[time_col].dropna()
        if values.empty:
            return None, None
        return str(values.min()), str(values.max())

    def _files(frame: pd.DataFrame) -> list[str]:
        if file_col is None:
            return [f"{name} (in-memory)"]
        return [str(v) for v in pd.unique(frame[file_col].dropna())]

    def _rows(frame: pd.DataFrame) -> list[int]:
        if record_col is None:
            return []
        return frame[record_col].head(PROVENANCE_SAMPLE_SIZE).tolist()

    # ---- host nodes ------------------------------------------------------
    host_cols = {"src": src_col, "dst": dst_col}
    host_stats: dict[str, dict[str, Any]] = {}
    if has_ips:
        for role, column in host_cols.items():
            seen = events[column].dropna().astype(str)
            for ip, count in seen.value_counts().items():
                entry = host_stats.setdefault(str(ip), {"n": 0, "roles": set()})
                entry["n"] += int(count)
                entry["roles"].add(role)

    if has_ips:
        for ip, entry in host_stats.items():
            first_seen, last_seen = (None, None)
            if has_timestamps:
                subset = events.loc[
                    (events[src_col].astype(str) == ip) | (events[dst_col].astype(str) == ip)
                ]
                first_seen, last_seen = _times(subset)
            pid = kg.add_provenance(
                created_by="build_knowledge_graph.host_nodes",
                files=sorted({f for role in entry["roles"] for f in _files(events)}),
                record_sample=_rows(events),
                n_supporting_records=entry["n"],
            )
            kg.add_node(
                f"ip:{ip}",
                "ip",
                label=ip,
                is_internal=is_private_ip(ip),
                evidence_count=entry["n"],
                confidence=OBSERVED_CONFIDENCE,
                confidence_rule="observed",
                first_seen=first_seen,
                last_seen=last_seen,
                roles="|".join(sorted(entry["roles"])),
                provenance_id=pid,
            )

    # Internal addresses also become host entities, linked by ALIAS_OF.
    if has_ips:
        internal_ips = [f"ip:{ip}" for ip in host_stats if is_private_ip(ip)]
        if internal_ips:
            pid = kg.add_provenance(
                created_by="build_knowledge_graph.host_alias_edges",
                files=_files(events),
                record_sample=_rows(events),
                n_supporting_records=len(internal_ips),
            )
            for ip_node in internal_ips:
                address = ip_node.split(":", 1)[1]
                kg.add_node(
                    f"host:{address}", "host", label=f"host({address})",
                    is_internal=True, evidence_count=host_stats[address]["n"],
                    confidence=OBSERVED_CONFIDENCE, confidence_rule="observed", provenance_id=pid,
                )
                kg.add_edge(
                    ip_node, f"host:{address}", "ALIAS_OF",
                    evidence_count=host_stats[address]["n"], confidence=OBSERVED_CONFIDENCE,
                    confidence_rule="observed", provenance_id=pid,
                    basis="RFC1918 address mapped to a host entity (1:1 in this prototype)",
                )

    # ---- service and protocol nodes --------------------------------------
    if dport_col is not None:
        service_frame = events[[dport_col] + ([proto_col] if proto_col else [])].dropna(subset=[dport_col])
        if not service_frame.empty:
            service_counts = service_frame.groupby(dport_col, dropna=True).size()
            pid = kg.add_provenance(
                created_by="build_knowledge_graph.service_nodes",
                files=_files(events), record_sample=_rows(events), n_supporting_records=int(service_counts.sum()),
            )
            for port, count in service_counts.items():
                service_label = port_to_service(port)
                kg.add_node(
                    f"svc:{service_label}/{int(port)}", "service",
                    label=f"{service_label}/{int(port)}", port=int(port),
                    service_name=service_label, evidence_count=int(count),
                    confidence=OBSERVED_CONFIDENCE, confidence_rule="observed", provenance_id=pid,
                )
    if proto_col is not None:
        proto_counts = events[proto_col].dropna().value_counts()
        pid = kg.add_provenance(
            created_by="build_knowledge_graph.protocol_nodes",
            files=_files(events), record_sample=_rows(events), n_supporting_records=int(proto_counts.sum()),
        )
        for protocol, count in proto_counts.items():
            kg.add_node(
                f"proto:{protocol}", "protocol", label=f"protocol({protocol})",
                protocol=int(protocol), evidence_count=int(count),
                confidence=OBSERVED_CONFIDENCE, confidence_rule="observed", provenance_id=pid,
            )

    # ---- aggregated host-to-host relationships ---------------------------
    if has_ips:
        pair_columns = [src_col, dst_col]
        ordered = events.dropna(subset=pair_columns).copy()
        if not ordered.empty:
            ordered["__src"] = ordered[src_col].astype(str)
            ordered["__dst"] = ordered[dst_col].astype(str)
            aggregations: dict[str, Any] = {"n_flows": ("__dst", "size")}
            bytes_cols = [c for c in ("Total Length of Fwd Packets", "Total Length of Bwd Packets") if c in ordered.columns]
            if bytes_cols:
                ordered["__bytes"] = ordered[bytes_cols].sum(axis=1)
                aggregations["bytes_total"] = ("__bytes", "sum")
            if time_col is not None and has_timestamps:
                aggregations["first_seen"] = (time_col, "min")
                aggregations["last_seen"] = (time_col, "max")
            grouped = ordered.groupby(["__src", "__dst"], sort=False).agg(**aggregations)
            sample_rows = ordered.groupby(["__src", "__dst"], sort=False)[
                "__record_id" if record_col else "__src"
            ].apply(lambda s: s.head(PROVENANCE_SAMPLE_SIZE).tolist())
            sample_files = ordered.groupby(["__src", "__dst"], sort=False)[
                file_col if file_col else "__src"
            ].apply(lambda s: sorted(set(str(v) for v in s.head(20))))

            k = float(settings.graph.derived_confidence_k)
            for (source, target), row in grouped.iterrows():
                n_flows = int(row["n_flows"])
                record_sample = (
                    [int(r) for r in sample_rows.loc[(source, target)]] if record_col else []
                )
                pid = kg.add_provenance(
                    created_by="build_knowledge_graph.communicates_with_edges",
                    files=list(sample_files.loc[(source, target)]),
                    record_sample=record_sample,
                    n_supporting_records=n_flows,
                )
                attributes: dict[str, Any] = {
                    "evidence_count": n_flows,
                    "confidence": round(aggregated_confidence(n_flows, k), 6),
                    "confidence_rule": f"aggregated: 1 - exp(-n/{k:g})",
                    "provenance_id": pid,
                }
                if "bytes_total" in row and not pd.isna(row["bytes_total"]):
                    attributes["bytes_total"] = float(row["bytes_total"])
                for key in ("first_seen", "last_seen"):
                    if key in row and row[key] is not None and not pd.isna(row[key]):
                        attributes[key] = str(row[key])
                kg.add_edge(f"ip:{source}", f"ip:{target}", "COMMUNICATES_WITH", **attributes)

    # ---- IP -> service / protocol usage ----------------------------------
    if has_ips and dport_col is not None:
        usage = events.dropna(subset=[dport_col]).copy()
        if not usage.empty:
            usage["__svc"] = [
                f"svc:{port_to_service(p)}/{int(p)}" for p in usage[dport_col]
            ]
            for ip_role, ip_col_nm in (("Source IP", src_col), ("Destination IP", dst_col)):
                pairs = usage.dropna(subset=[ip_col_nm])
                if pairs.empty:
                    continue
                grouped_usage = pairs.groupby([pairs[ip_col_nm].astype(str), "__svc"], sort=False).size()
                pid = kg.add_provenance(
                    created_by=f"build_knowledge_graph.uses_service_edges[{ip_role}]",
                    files=_files(events), record_sample=_rows(events), n_supporting_records=int(grouped_usage.sum()),
                )
                k = float(settings.graph.derived_confidence_k)
                for (ip, service), count in grouped_usage.items():
                    node_id = f"ip:{ip}"
                    if node_id not in kg.graph or service not in kg.graph:
                        continue
                    kg.add_edge(
                        node_id, service, "USES_SERVICE",
                        evidence_count=int(count),
                        confidence=round(aggregated_confidence(int(count), k), 6),
                        confidence_rule=f"aggregated: 1 - exp(-n/{k:g})",
                        provenance_id=pid,
                    )

    if dport_col is not None and proto_col is not None:
        service_proto = events.dropna(subset=[dport_col, proto_col]).copy()
        if not service_proto.empty:
            service_proto["__svc"] = [f"svc:{port_to_service(p)}/{int(p)}" for p in service_proto[dport_col]]
            grouped_sp = service_proto.groupby(["__svc", proto_col], sort=False).size()
            pid = kg.add_provenance(
                created_by="build_knowledge_graph.runs_on_protocol_edges",
                files=_files(events), record_sample=_rows(events), n_supporting_records=int(grouped_sp.sum()),
            )
            k = float(settings.graph.derived_confidence_k)
            for (service, protocol), count in grouped_sp.items():
                if service not in kg.graph or f"proto:{protocol}" not in kg.graph:
                    continue
                kg.add_edge(
                    service, f"proto:{protocol}", "RUNS_ON_PROTOCOL",
                    evidence_count=int(count),
                    confidence=round(aggregated_confidence(int(count), k), 6),
                    confidence_rule=f"aggregated: 1 - exp(-n/{k:g})",
                    provenance_id=pid,
                )

    # ---- event (SecurityEvent) nodes -------------------------------------
    if settings.graph.include_event_nodes:
        n_events = min(len(events), int(settings.graph.max_event_nodes))
        if n_events < len(events):
            kg.build_log.append(
                f"event nodes limited to {n_events} of {len(events)} records "
                "(graph.include_event_nodes / graph.max_event_nodes)"
            )
        event_slice = events.head(n_events)
        event_pid = kg.add_provenance(
            created_by="build_knowledge_graph.event_nodes",
            files=_files(event_slice), record_sample=_rows(event_slice), n_supporting_records=n_events,
        )
        payload = event_slice[[c for c in (record_col, time_col, file_col) if c is not None]].to_dict("records")
        time_lookup = {
            int(record["__record_id"]): record.get(time_col)
            for record in payload
            if record_col is not None and record.get("__record_id") is not None
        }
        for index in event_slice.index:
            rid = int(event_slice.at[index, record_col]) if record_col else int(index)
            node_id = f"evt:{rid}"
            first_seen = time_lookup.get(rid)
            if first_seen is not None and pd.isna(first_seen):
                first_seen = None
            kg.add_node(
                node_id, "event", label=f"event#{rid}",
                evidence_count=1, confidence=OBSERVED_CONFIDENCE, confidence_rule="observed",
                first_seen=None if first_seen is None else str(first_seen),
                provenance_id=event_pid,
            )
            service_node = None
            if dport_col is not None and not pd.isna(event_slice.at[index, dport_col]):
                service_node = f"svc:{port_to_service(event_slice.at[index, dport_col])}/{int(event_slice.at[index, dport_col])}"
            if has_ips:
                source_ip = event_slice.at[index, src_col]
                target_ip = event_slice.at[index, dst_col]
                if not pd.isna(source_ip) and f"ip:{source_ip}" in kg.graph:
                    kg.add_edge(node_id, f"ip:{source_ip}", "INITIATED_BY",
                                evidence_count=1, confidence=OBSERVED_CONFIDENCE,
                                confidence_rule="observed", provenance_id=event_pid)
                if not pd.isna(target_ip) and f"ip:{target_ip}" in kg.graph:
                    kg.add_edge(node_id, f"ip:{target_ip}", "TARGETED",
                                evidence_count=1, confidence=OBSERVED_CONFIDENCE,
                                confidence_rule="observed", provenance_id=event_pid)
            if service_node and service_node in kg.graph:
                kg.add_edge(node_id, service_node, "USES_SERVICE",
                            evidence_count=1, confidence=OBSERVED_CONFIDENCE,
                            confidence_rule="observed", provenance_id=event_pid)

    # ---- technique nodes (labels required; never a model input) ----------
    if include_labels and CATEGORY_COLUMN in events.columns and record_col is not None:
        labelled = events.loc[events[CATEGORY_COLUMN].astype(str) != "Benign"]
        if not labelled.empty:
            tech_pid = kg.add_provenance(
                created_by="build_knowledge_graph.technique_nodes",
                files=_files(labelled), record_sample=_rows(labelled), n_supporting_records=int(len(labelled)),
            )
            technique_counts = labelled[CATEGORY_COLUMN].astype(str).value_counts()
            for technique, count in technique_counts.items():
                kg.add_node(
                    f"tech:{technique}", "technique", label=technique,
                    evidence_count=int(count), confidence=OBSERVED_CONFIDENCE,
                    confidence_rule="observed", provenance_id=tech_pid,
                    note="label-derived; the label-to-technique mapping is ours (see docs/DATASET.md)",
                )
            if settings.graph.include_event_nodes:
                capped = labelled.head(int(settings.graph.max_event_nodes))
                for index, technique in zip(capped.index, capped[CATEGORY_COLUMN].astype(str)):
                    rid = int(capped.at[index, record_col])
                    node_id = f"evt:{rid}"
                    if node_id in kg.graph and f"tech:{technique}" in kg.graph:
                        kg.add_edge(node_id, f"tech:{technique}", "MATCHES_TECHNIQUE",
                                    evidence_count=1, confidence=OBSERVED_CONFIDENCE,
                                    confidence_rule="observed", provenance_id=tech_pid)
                kg.build_log.append(
                    "MATCHES_TECHNIQUE edges come from dataset labels and are used for human "
                    "inspection / explanation only - they are never model features."
                )

    # ---- threat-intel nodes ----------------------------------------------
    if intel_index and intel_index.get("available"):
        flagged = [ip for ip in intel_index["flagged_ips"] if f"ip:{ip}" in kg.graph]
        if flagged:
            intel_pid = kg.add_provenance(
                created_by="build_knowledge_graph.intel_nodes",
                files=[f"{name} (training split labels)"], record_sample=[],
                n_supporting_records=len(flagged),
            )
            for ip in flagged:
                info = intel_index["ips"].get(ip, {})
                confidence = float(info.get("confidence", intel_confidence(0, 0)))
                kg.add_node(
                    f"intel:{ip}", "intel", label=f"training-labelled malicious ({ip})",
                    evidence_count=int(info.get("n_malicious_records", 0)),
                    confidence=confidence,
                    confidence_rule="intel_smoothed_ratio: (n_malicious + 1) / (n_total + 2)",
                    provenance_id=intel_pid,
                    source=intel_index.get("source", "training labels"),
                )
                kg.add_edge(
                    f"ip:{ip}", f"intel:{ip}", "FLAGGED_AS_INTEL",
                    evidence_count=int(info.get("n_malicious_records", 0)),
                    confidence=confidence,
                    confidence_rule="intel_smoothed_ratio: (n_malicious + 1) / (n_total + 2)",
                    provenance_id=intel_pid,
                    basis="IP appeared in at least one training-labelled malicious record",
                )
            kg.build_log.append(
                f"{len(flagged)} IP(s) carry training-derived threat-intel evidence; "
                "this is a hindsight index over training labels, not an external feed."
            )

    logger.info(
        "graph '%s': %d nodes / %d edges | %s",
        name, kg.graph.number_of_nodes(), kg.graph.number_of_edges(),
        ", ".join(f"{k}={v}" for k, v in sorted(kg.node_type_counts().items())),
    )
    return kg
