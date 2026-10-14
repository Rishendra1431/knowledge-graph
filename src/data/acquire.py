"""Data acquisition: local CSVs, optional download, documented synthetic fallback.

The prototype never invents a dataset silently. ``acquire_dataset`` returns the
frame *and* a provenance record that travels with every result artifact, so a
synthetic run can never be mistaken for a finding about real traffic.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, Settings, get_logger

logger = get_logger("data.acquire")

# ---------------------------------------------------------------------------
# Canonical CICIDS2017 schema (CICFlowMeter 2.0, MachineLearningCSV release).
# Column names in the released CSVs carry a leading space, which
# ``normalize_columns`` strips.
# ---------------------------------------------------------------------------
CICIDS2017_COLUMNS: list[str] = [
    "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port",
    "Protocol", "Timestamp", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets",
    "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min",
    "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max",
    "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
    "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max",
    "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
    "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
    "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
    "Avg Bwd Segment Size", "Fwd Header Length.1", "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "Label",
]

# Released CICIDS2017 label values, grouped into the coarser categories used for
# reporting. The mapping is ours (documented in docs/DATASET.md), not the
# dataset's: it only collapses fine-grained labels, it never changes a
# BENIGN/attack decision.
ATTACK_TAXONOMY: dict[str, str] = {
    "BENIGN": "Benign",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "Heartbleed": "Heartbleed",
    "DDoS": "DDoS",
    "PortScan": "PortScan",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack \u2013 Brute Force": "WebAttack",
    "Web Attack \u2013 XSS": "WebAttack",
    "Web Attack \u2013 Sql Injection": "WebAttack",
    "Web Attack - Brute Force": "WebAttack",  # hyphen variants seen in the wild
    "Web Attack - XSS": "WebAttack",
    "Web Attack - Sql Injection": "WebAttack",
    "Infiltration": "Infiltration",
    "Bot": "Bot",
}

PROVENANCE_COLUMNS = ["__source_file", "__source_row", "__dataset_origin"]


@dataclass
class DatasetProvenance:
    """Everything a reader needs to judge where the data came from."""

    name: str
    origin: str  # "local_csv" | "downloaded" | "synthetic"
    files: list[str] = field(default_factory=list)
    file_hashes: dict[str, str] = field(default_factory=dict)
    raw_rows: int = 0
    rows_loaded: int = 0
    columns: list[str] = field(default_factory=list)
    label_values: dict[str, int] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    retrieved_at_utc: str = ""

    @property
    def is_synthetic(self) -> bool:
        return self.origin == "synthetic"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "origin": self.origin,
            "is_synthetic": self.is_synthetic,
            "files": self.files,
            "file_hashes_sha256": self.file_hashes,
            "raw_rows": self.raw_rows,
            "rows_loaded": self.rows_loaded,
            "columns": self.columns,
            "label_values": self.label_values,
            "sampling": self.sampling,
            "notes": self.notes,
            "retrieved_at_utc": self.retrieved_at_utc,
        }
        return payload


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Strip whitespace from column names and drop positional duplicates.

    Returns the frame plus the list of duplicate column names that were dropped.
    ``Fwd Header Length.1`` is a known artefact of CICFlowMeter and is retained
    here (it is listed in ``preprocess.drop_columns``); only *identically named*
    duplicates are removed, keeping the first occurrence.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    seen: dict[str, int] = {}
    keep_idx: list[int] = []
    dropped: list[str] = []
    for idx, name in enumerate(df.columns):
        if name in seen:
            dropped.append(name)
            continue
        seen[name] = idx
        keep_idx.append(idx)
    if dropped:
        logger.warning("dropping duplicate column names: %s", sorted(set(dropped)))
        df = df.iloc[:, keep_idx]
    return df, dropped


def _strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    return df


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv_robust(path: Path, nrows: int | None, sample_strategy: str) -> tuple[pd.DataFrame, int]:
    """Read one CSV, tolerating the mixed dtypes and bad tokens in this dataset."""
    read_kwargs: dict[str, Any] = {
        "low_memory": False,
        "na_values": ["", " ", "NaN", "nan", "NULL", "Infinity", "inf", "-inf", "Inf"],
        "keep_default_na": True,
    }
    try:
        total_rows = sum(1 for _ in path.open("rb")) - 1  # header
    except OSError:
        total_rows = -1

    if nrows is not None and total_rows > nrows:
        if sample_strategy == "random":
            # Deterministic spread sample: read in chunks and keep every k-th row.
            step = max(total_rows // nrows, 1)
            chunks = []
            collected = 0
            for chunk in pd.read_csv(path, chunksize=20000, **read_kwargs):
                take = chunk.iloc[::step]
                chunks.append(take)
                collected += len(take)
            frame = pd.concat(chunks, ignore_index=True)
            if len(frame) > nrows:
                # Evenly spaced selection keeps ordering information intact.
                positions = np.linspace(0, len(frame) - 1, nrows).astype(int)
                frame = frame.iloc[positions]
        else:  # "head"
            frame = pd.read_csv(path, nrows=nrows, **read_kwargs)
    else:
        frame = pd.read_csv(path, **read_kwargs)

    frame["__source_row"] = np.arange(len(frame), dtype=np.int64)
    return frame, max(total_rows, 0)


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(settings: Settings) -> tuple[pd.DataFrame, DatasetProvenance]:
    """Build a *synthetic* labelled flow dataset with the CICIDS2017 schema.

    Purpose: keep the pipeline executable (and unit-testable) when the real
    dataset has not been obtained. It is deliberately documented as synthetic
    everywhere downstream and in the results artifact. Behaviour patterns are
    hand-designed (port scan fan-out, DoS packet rates, repeated auth attempts,
    periodic beaconing), so any model performance on it measures *our
    generator*, not real-world detection ability.
    """
    rng = np.random.default_rng(settings.seed)
    n = int(settings.dataset.synthetic_rows)

    internal = [f"192.168.10.{i}" for i in range(2, 40)]
    servers = ["205.174.165.80", "205.174.165.73", "205.174.165.68"]
    external = [f"172.16.{rng.integers(0, 255)}.{rng.integers(1, 254)}" for _ in range(60)]
    attackers = ["45.118.132.11", "45.118.132.12", "13.58.98.5", "104.211.44.9"]

    scenario_weights = {
        "benign": 0.62,
        "PortScan": 0.10,
        "DoS": 0.10,
        "DDoS": 0.05,
        "BruteForce": 0.06,
        "WebAttack": 0.04,
        "Bot": 0.03,
    }
    scenarios = rng.choice(
        list(scenario_weights), size=n, p=list(scenario_weights.values())
    )

    start = datetime(2017, 7, 3, 8, 0, 0)  # Monday, mirrors the real capture week
    day_offsets = np.array([0, 1, 2, 3, 4])
    day = rng.choice(day_offsets, size=n, p=[0.20, 0.20, 0.18, 0.17, 0.25])
    seconds = rng.integers(0, 8 * 3600, size=n)
    timestamps = [start + timedelta(days=int(d), seconds=int(s)) for d, s in zip(day, seconds)]

    rows: list[dict[str, Any]] = []
    for i in range(n):
        scenario = str(scenarios[i])
        src, dst, sport, dport, proto = _sample_endpoints(rng, scenario, internal, servers, external, attackers)

        # Baselines (approximate CICFlowMeter magnitudes, not fitted to the real data)
        duration_us = float(max(rng.lognormal(mean=11.0, sigma=1.6), 1.0))
        fwd_packets = float(max(rng.lognormal(mean=1.8, sigma=0.9), 1.0))
        bwd_packets = float(max(rng.lognormal(mean=1.6, sigma=1.0), 0.0))
        fwd_bytes = fwd_packets * float(rng.uniform(40, 900))
        bwd_bytes = bwd_packets * float(rng.uniform(40, 1200))

        if scenario in ("DoS", "DDoS"):
            duration_us = float(rng.uniform(500, 60_000))
            fwd_packets = float(rng.uniform(300, 4000))
            fwd_bytes = fwd_packets * float(rng.uniform(40, 90))
            bwd_packets = float(rng.uniform(0, 20))
            bwd_bytes = bwd_packets * float(rng.uniform(40, 100))
        elif scenario == "PortScan":
            duration_us = float(rng.uniform(1, 2000))
            fwd_packets = float(rng.integers(1, 3))
            fwd_bytes = fwd_packets * 40.0
            # SYN with no completed handshake
            bwd_packets = float(rng.integers(0, 2))
            bwd_bytes = bwd_packets * 40.0
        elif scenario in ("BruteForce", "WebAttack"):
            duration_us = float(rng.uniform(1000, 250_000))
            fwd_packets = float(rng.integers(3, 40))
            fwd_bytes = fwd_packets * float(rng.uniform(60, 600))
            bwd_packets = float(rng.integers(2, 40))
            bwd_bytes = bwd_packets * float(rng.uniform(60, 800))
        elif scenario == "Bot":
            duration_us = float(rng.uniform(2000, 20_000))
            fwd_packets = float(rng.integers(2, 12))
            fwd_bytes = fwd_packets * float(rng.uniform(80, 300))
            bwd_packets = float(rng.integers(2, 12))
            bwd_bytes = bwd_packets * float(rng.uniform(80, 300))

        flow_bytes_s = fwd_bytes + bwd_bytes
        flow_packets_s = fwd_packets + bwd_packets
        duration_s = duration_us / 1e6
        flow_bytes_rate = flow_bytes_s / duration_s if duration_s > 0 else 0.0
        flow_packets_rate = flow_packets_s / duration_s if duration_s > 0 else 0.0

        syn = 1.0 if scenario == "PortScan" else float(rng.integers(0, 2))
        ack = 0.0 if scenario == "PortScan" else float(rng.integers(0, 2))
        psh = 1.0 if scenario in ("WebAttack", "BruteForce", "Bot") else 0.0
        fin = float(rng.integers(0, 2)) if scenario == "benign" else 0.0
        rst = 1.0 if scenario == "PortScan" and rng.random() < 0.4 else 0.0

        row = {
            "Flow ID": f"{src}-{dst}-{sport}-{dport}-{proto}",
            "Source IP": src,
            "Source Port": int(sport),
            "Destination IP": dst,
            "Destination Port": int(dport),
            "Protocol": int(proto),
            "Timestamp": timestamps[i],
            "Flow Duration": duration_us,
            "Total Fwd Packets": fwd_packets,
            "Total Backward Packets": bwd_packets,
            "Total Length of Fwd Packets": fwd_bytes,
            "Total Length of Bwd Packets": bwd_bytes,
            "Fwd Packet Length Max": fwd_bytes / max(fwd_packets, 1) * 1.3,
            "Fwd Packet Length Min": fwd_bytes / max(fwd_packets, 1) * 0.6,
            "Fwd Packet Length Mean": fwd_bytes / max(fwd_packets, 1),
            "Fwd Packet Length Std": fwd_bytes / max(fwd_packets, 1) * 0.2,
            "Bwd Packet Length Max": bwd_bytes / max(bwd_packets, 1) * 1.3,
            "Bwd Packet Length Min": bwd_bytes / max(bwd_packets, 1) * 0.6,
            "Bwd Packet Length Mean": bwd_bytes / max(bwd_packets, 1),
            "Bwd Packet Length Std": bwd_bytes / max(bwd_packets, 1) * 0.2,
            "Flow Bytes/s": flow_bytes_rate,
            "Flow Packets/s": flow_packets_rate,
            "Flow IAT Mean": duration_us / max(flow_packets_s, 1),
            "Flow IAT Std": duration_us / max(flow_packets_s, 1) * 0.5,
            "Flow IAT Max": duration_us / max(flow_packets_s, 1) * 2.0,
            "Flow IAT Min": duration_us / max(flow_packets_s, 1) * 0.1,
            "Fwd IAT Total": duration_us,
            "Fwd IAT Mean": duration_us / max(fwd_packets, 1),
            "Fwd IAT Std": duration_us / max(fwd_packets, 1) * 0.4,
            "Fwd IAT Max": duration_us,
            "Fwd IAT Min": duration_us / max(fwd_packets, 1) * 0.1,
            "Bwd IAT Total": duration_us if bwd_packets > 0 else 0.0,
            "Bwd IAT Mean": duration_us / max(bwd_packets, 1),
            "Bwd IAT Std": duration_us / max(bwd_packets, 1) * 0.4,
            "Bwd IAT Max": duration_us,
            "Bwd IAT Min": duration_us / max(bwd_packets, 1) * 0.1,
            "Fwd PSH Flags": psh,
            "Bwd PSH Flags": 1.0 if scenario == "benign" else 0.0,
            "Fwd URG Flags": 0.0,
            "Bwd URG Flags": 0.0,
            "Fwd Header Length": 20.0 * fwd_packets,
            "Bwd Header Length": 20.0 * bwd_packets,
            "Fwd Packets/s": fwd_packets / max(duration_s, 1e-6),
            "Bwd Packets/s": bwd_packets / max(duration_s, 1e-6),
            "Min Packet Length": 0.0 if scenario == "PortScan" else 40.0,
            "Max Packet Length": max(fwd_bytes / max(fwd_packets, 1), 40.0),
            "Packet Length Mean": flow_bytes_s / max(flow_packets_s, 1),
            "Packet Length Std": flow_bytes_s / max(flow_packets_s, 1) * 0.3,
            "Packet Length Variance": (flow_bytes_s / max(flow_packets_s, 1)) ** 2 * 0.09,
            "FIN Flag Count": fin,
            "SYN Flag Count": syn,
            "RST Flag Count": rst,
            "PSH Flag Count": psh,
            "ACK Flag Count": ack,
            "URG Flag Count": 0.0,
            "CWE Flag Count": 0.0,
            "ECE Flag Count": 0.0,
            "Down/Up Ratio": bwd_packets / max(fwd_packets, 1),
            "Average Packet Size": flow_bytes_s / max(flow_packets_s, 1),
            "Avg Fwd Segment Size": fwd_bytes / max(fwd_packets, 1),
            "Avg Bwd Segment Size": bwd_bytes / max(bwd_packets, 1),
            "Fwd Header Length.1": 20.0 * fwd_packets,
            "Fwd Avg Bytes/Bulk": 0.0,
            "Fwd Avg Packets/Bulk": 0.0,
            "Fwd Avg Bulk Rate": 0.0,
            "Bwd Avg Bytes/Bulk": 0.0,
            "Bwd Avg Packets/Bulk": 0.0,
            "Bwd Avg Bulk Rate": 0.0,
            "Subflow Fwd Packets": fwd_packets,
            "Subflow Fwd Bytes": fwd_bytes,
            "Subflow Bwd Packets": bwd_packets,
            "Subflow Bwd Bytes": bwd_bytes,
            "Init_Win_bytes_forward": float(rng.integers(0, 65535)),
            "Init_Win_bytes_backward": float(rng.integers(0, 65535)),
            "act_data_pkt_fwd": float(rng.integers(0, max(int(fwd_packets), 2))),
            "min_seg_size_forward": 20.0,
            "Active Mean": duration_us / 4.0,
            "Active Std": duration_us / 20.0,
            "Active Max": duration_us / 2.0,
            "Active Min": duration_us / 10.0,
            "Idle Mean": duration_us / 3.0 if scenario == "Bot" else 0.0,
            "Idle Std": 0.0,
            "Idle Max": duration_us / 2.0 if scenario == "Bot" else 0.0,
            "Idle Min": 0.0,
            "Label": "BENIGN" if scenario == "benign" else _synthetic_label(rng, scenario),
        }
        rows.append(row)

    frame = pd.DataFrame(rows)

    # Deliberately inject the same defects the real dataset contains so the
    # cleaning stage is genuinely exercised.
    n_inf = max(int(n * 0.004), 1)
    inf_rows = rng.choice(n, size=n_inf, replace=False)
    frame.loc[inf_rows, "Flow Bytes/s"] = np.inf
    frame.loc[inf_rows, "Flow Packets/s"] = np.inf

    n_nan = max(int(n * 0.003), 1)
    for col in ("Flow IAT Mean", "Bwd Packet Length Mean", "Idle Mean"):
        nan_rows = rng.choice(n, size=n_nan, replace=False)
        frame.loc[nan_rows, col] = np.nan

    # A block of exact duplicates (the real files contain thousands).
    dup_source = rng.choice(n, size=max(int(n * 0.01), 1), replace=False)
    frame = pd.concat([frame, frame.iloc[dup_source]], ignore_index=True)
    frame = frame.sample(frac=1.0, random_state=settings.seed).reset_index(drop=True)
    frame["__source_row"] = np.arange(len(frame), dtype=np.int64)

    label_counts = frame["Label"].value_counts().to_dict()
    provenance = DatasetProvenance(
        name=f"{settings.dataset.name}-SYNTHETIC",
        origin="synthetic",
        files=[],
        file_hashes={},
        raw_rows=len(frame),
        rows_loaded=len(frame),
        columns=[c for c in frame.columns if c not in PROVENANCE_COLUMNS],
        label_values={str(k): int(v) for k, v in label_counts.items()},
        sampling={"strategy": "generated", "requested_rows": n, "rows_generated": len(frame)},
        notes=[
            "SYNTHETIC DATA. No real network traffic was collected.",
            "Generated by src/data/acquire.py::generate_synthetic_dataset with a seeded RNG.",
            "Feature distributions are hand-designed, so model scores reflect the generator, "
            "not real-world detection capability.",
            "Injected defects: inf rates, NaNs, exact duplicate rows, week-spanning timestamps.",
        ],
        retrieved_at_utc=datetime.utcnow().isoformat() + "Z",
    )
    logger.warning(
        "generated %d SYNTHETIC flow rows (origin=synthetic); results must be labelled synthetic",
        len(frame),
    )
    return frame, provenance


def _synthetic_label(rng: np.random.Generator, scenario: str) -> str:
    if scenario == "DoS":
        return str(rng.choice(["DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest"]))
    if scenario == "BruteForce":
        return str(rng.choice(["FTP-Patator", "SSH-Patator"]))
    if scenario == "WebAttack":
        return str(
            rng.choice(
                [
                    "Web Attack \u2013 Brute Force",
                    "Web Attack \u2013 XSS",
                    "Web Attack \u2013 Sql Injection",
                ]
            )
        )
    return scenario


def _sample_endpoints(
    rng: np.random.Generator,
    scenario: str,
    internal: list[str],
    servers: list[str],
    external: list[str],
    attackers: list[str],
) -> tuple[str, str, int, int, int]:
    """Pick endpoints consistent with the scenario (direction and fan-out)."""
    if scenario == "benign":
        internal_src = rng.random() < 0.7
        if internal_src:
            return (
                str(rng.choice(internal)),
                str(rng.choice(external)),
                int(rng.integers(40000, 65000)),
                int(rng.choice([80, 443, 53, 443, 8080])),
                6 if rng.random() < 0.9 else 17,
            )
        return (
            str(rng.choice(external)),
            str(rng.choice(servers)),
            int(rng.integers(30000, 65000)),
            int(rng.choice([80, 443])),
            6,
        )
    attacker = str(rng.choice(attackers))
    if scenario == "PortScan":
        return attacker, str(rng.choice(servers)), int(rng.integers(30000, 65000)), int(rng.integers(1, 1024)), 6
    if scenario in ("DoS", "DDoS"):
        return attacker, str(rng.choice(servers)), int(rng.integers(30000, 65000)), 80, 6
    if scenario == "BruteForce":
        return str(rng.choice(internal)), str(rng.choice(servers)), int(rng.integers(30000, 65000)), int(rng.choice([21, 22, 3389])), 6
    if scenario == "WebAttack":
        return str(rng.choice([*internal, *external])), str(rng.choice(servers)), int(rng.integers(30000, 65000)), int(rng.choice([80, 443])), 6
    # Bot: internal host beaconing to an external C2
    return str(rng.choice(internal)), attacker, int(rng.integers(30000, 65000)), int(rng.choice([8080, 443, 6667])), 6


# ---------------------------------------------------------------------------
# Acquisition entry point
# ---------------------------------------------------------------------------
def find_raw_files(settings: Settings) -> list[Path]:
    raw_dir = (PROJECT_ROOT / settings.dataset.raw_dir).resolve()
    if not raw_dir.exists():
        return []
    files = sorted(p for p in raw_dir.rglob("*.csv") if p.is_file())
    return files


def maybe_download(settings: Settings) -> list[Path]:
    """Best-effort download of the configured URLs. Never raises."""
    if not settings.dataset.download_urls:
        logger.info("no dataset.download_urls configured; skipping download step")
        return []
    raw_dir = (PROJECT_ROOT / settings.dataset.raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    try:
        import requests
    except ImportError:
        logger.warning("requests not installed; cannot download the dataset")
        return []

    downloaded: list[Path] = []
    for url in settings.dataset.download_urls:
        target = raw_dir / Path(url.split("?")[0]).name
        if target.exists():
            downloaded.append(target)
            continue
        try:
            logger.info("downloading %s -> %s", url, target.name)
            with requests.get(url, stream=True, timeout=settings.dataset.download_timeout_seconds) as response:
                response.raise_for_status()
                with target.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        handle.write(chunk)
            downloaded.append(target)
        except Exception as exc:  # network problems must not kill the pipeline
            logger.warning("download failed for %s: %s", url, exc)
    return downloaded


def acquire_dataset(settings: Settings) -> tuple[pd.DataFrame, DatasetProvenance]:
    """Load the dataset from disk, or generate the documented synthetic fallback."""
    files = find_raw_files(settings)
    if not files and settings.dataset.download_urls:
        maybe_download(settings)
        files = find_raw_files(settings)

    if not files:
        if not settings.dataset.allow_synthetic_fallback:
            raise FileNotFoundError(
                f"no CSV files found in {settings.path('raw_dir')} and "
                "dataset.allow_synthetic_fallback is false. See docs/DATASET.md."
            )
        logger.warning("no raw CSVs found in %s", (PROJECT_ROOT / settings.dataset.raw_dir).resolve())
        return generate_synthetic_dataset(settings)

    frames: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    total_raw = 0
    for path in files:
        frame, raw_rows = _read_csv_robust(path, settings.dataset.max_rows_per_file, settings.dataset.sample_strategy)
        frame, _ = normalize_columns(frame)
        frame["__source_file"] = path.name
        frames.append(frame)
        total_raw += raw_rows
        try:
            hashes[path.name] = _sha256(path)
        except OSError:
            hashes[path.name] = "unavailable"
        logger.info("read %s: %d rows loaded of %d", path.name, len(frame), raw_rows)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["__dataset_origin"] = "real_csv"

    label_col = settings.preprocess.label_column
    label_counts: dict[str, int] = {}
    if label_col in combined.columns:
        stripped = combined[label_col].astype(str).str.strip()
        label_counts = {str(k): int(v) for k, v in stripped.value_counts().to_dict().items()}

    provisioning_note = []
    if settings.dataset.max_rows_per_file:
        provisioning_note.append(
            f"prototype subset: at most {settings.dataset.max_rows_per_file} rows per file "
            f"(strategy={settings.dataset.sample_strategy}); full-dataset numbers will differ"
        )

    provenance = DatasetProvenance(
        name=settings.dataset.name,
        origin="local_csv",
        files=[p.name for p in files],
        file_hashes=hashes,
        raw_rows=total_raw,
        rows_loaded=len(combined),
        columns=[c for c in combined.columns if c not in PROVENANCE_COLUMNS],
        label_values=label_counts,
        sampling={
            "max_rows_per_file": settings.dataset.max_rows_per_file,
            "strategy": settings.dataset.sample_strategy,
            "raw_rows_on_disk": total_raw,
        },
        notes=provisioning_note,
        retrieved_at_utc=datetime.utcnow().isoformat() + "Z",
    )
    logger.info("dataset ready: %d rows from %d file(s)", len(combined), len(files))
    return combined, provenance
