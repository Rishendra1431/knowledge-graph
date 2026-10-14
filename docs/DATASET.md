# Dataset documentation

## 1. Primary dataset: CICIDS2017

**Full name:** Intrusion Detection Evaluation Dataset (CIC-IDS2017)
**Provider:** Canadian Institute for Cybersecurity (CIC), University of New Brunswick
**Landing page:** <https://www.unb.ca/cic/datasets/ids-2017.html>
**Format used here:** the `MachineLearningCSV` release (pre-extracted flow features, one CSV per capture day)

| Property | Value |
| --- | --- |
| Records (full dataset) | ~2.83 million labelled network flows |
| Capture window | Monday 3 July 2017 – Friday 7 July 2017 |
| Files | 8 CSVs (Thursday and Friday are split into morning/afternoon) |
| Features | 78 flow features extracted with CICFlowMeter, plus `Label` |
| Labels | `BENIGN` and 14 attack labels across 7 attack families |
| Class balance | heavily imbalanced: benign flows dominate the week |

### Files

```
Monday-WorkingHours.pcap_ISCX.csv
Tuesday-WorkingHours.pcap_ISCX.csv
Wednesday-workingHours.pcap_ISCX.csv
Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
Friday-WorkingHours-Morning.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

Put them in `data/raw/` (subdirectories are searched too). The loader reads every `*.csv`
under that directory, so you can start with one file.

### Labels and our taxonomy

The dataset's own labels are preserved in a `Label` column. A coarser `attack_category` column is
derived from them with the mapping in `src/data/acquire.py::ATTACK_TAXONOMY`:

| Dataset label | Our category |
| --- | --- |
| `BENIGN` | Benign |
| `DoS Hulk`, `DoS GoldenEye`, `DoS slowloris`, `DoS Slowhttptest` | DoS |
| `DDoS` | DDoS |
| `PortScan` | PortScan |
| `FTP-Patator`, `SSH-Patator` | BruteForce |
| `Web Attack – Brute Force`, `Web Attack – XSS`, `Web Attack – Sql Injection` | WebAttack |
| `Infiltration` | Infiltration |
| `Heartbleed` | Heartbleed |
| `Bot` | Bot |
| anything else | `Other` (and the unmapped value is logged) |

**This mapping is ours, not the dataset's.** It only collapses fine-grained labels for reporting;
it never changes a benign/attack decision. The binary target `is_attack` is `Label != 'BENIGN'`.

### Entity columns used by the knowledge graph

| Column | Used for |
| --- | --- |
| `Source IP`, `Destination IP` | `ip` nodes, `host` nodes (internal addresses), `COMMUNICATES_WITH` edges |
| `Source Port`, `Destination Port` | `service` nodes (well-known name lookup in `src/data/preprocess.py`) |
| `Protocol` | `protocol` nodes |
| `Timestamp` | day-based temporal split, node/edge `first_seen`/`last_seen` |

`Flow ID` is dropped: it is a per-record identifier that would let a model memorise rows.
`Fwd Header Length.1` is dropped: it is a duplicate artefact of CICFlowMeter (the `Fwd Header
Length` feature appears twice in the released CSVs).

### Entities the dataset cannot support (and which are therefore absent)

The problem statement mentions domains, malware, vulnerabilities and attack techniques. The
`MachineLearningCSV` release contains **no** DNS names, no file hashes, no CVE identifiers and no
MITRE ATT&CK mappings. Building such entities would mean inventing data, so the prototype does not
create them. `technique` nodes are only produced when `include_labels=True` is passed to the graph
builder (training/analysis view), using our label taxonomy as a stand-in, and they are never model
inputs. This is stated in `docs/LIMITATIONS.md` and in the graph summary of every run.

### Known defects handled by the pipeline

| Defect | Handling |
| --- | --- |
| Column names carry leading spaces | stripped in `normalize_columns` |
| Duplicate `Fwd Header Length` column | dropped via `preprocess.drop_columns` |
| `Infinity` / `NaN` in `Flow Bytes/s`, `Flow Packets/s`, IAT features | `inf` → `NaN`, then training-median imputation |
| Missing values in several columns | median imputation fitted on the training split; columns above `max_missing_fraction` are dropped |
| Thousands of exact duplicate rows | removed **before** splitting |
| Mixed numeric/string parsing | `pd.to_numeric(errors="coerce")`; the number of coerced values is reported |

### Prototype subset

The full dataset does not fit comfortably on a student laptop, so `config.yaml` limits
`dataset.max_rows_per_file` (default 25 000 rows per file ≈ 200 000 flows total). **Report the
subset size with your results**; metrics from a subset are not comparable to papers using all
2.83 million flows. Set `max_rows_per_file: 0` to use everything.

## 2. Synthetic fallback

If `data/raw/` is empty and `dataset.allow_synthetic_fallback: true`, the pipeline generates a
synthetic dataset (`src/data/acquire.py::generate_synthetic_dataset`) with the same 78-column
schema, week-spanning timestamps, injected `inf`/`NaN` values and duplicate rows.

It exists so the pipeline is runnable before the real data is obtained. When it is used:

- `dataset.is_synthetic` is `true` in `results/metrics.json`;
- the dashboard and the paper report show a prominent warning;
- the comparison claim is suffixed with "Evidence from a synthetic dataset…".

**Synthetic results measure the generator, not detection capability.** Never present them as
findings about real network traffic.

## 3. Alternative datasets

`config.yaml` targets CICIDS2017 out of the box, but the loader is generic. For another labelled
flow dataset, adjust:

```yaml
dataset:
  name: "UNSW-NB15"
preprocess:
  label_column: "attack_cat"     # or "Label"/"class"/"label"
  benign_label: "Normal"
  timestamp_column: "Timestamp"  # omit or point at the right column
  entity_columns: ["srcip", "dstip", "sport", "dsport", "proto"]
  temporal:
    train_days: ["Monday", "Tuesday", "Wednesday", "Thursday"]
    test_days: ["Friday"]
```

If no IP columns are found, the pipeline degrades gracefully: graph features are disabled, the
baseline is still evaluated, and the graph-aware arm is recorded as `not_executed` with the reason.
If the label column is missing, the run stops with a clear error.

## 4. Citation and usage terms

If you publish work using this dataset, cite the CIC publication:

> Iman Sharafaldin, Arash Habibi Lashkari, and Ali A. Ghorbani, "Toward Generating a New Intrusion
> Detection Dataset and Intrusion Traffic Characterization", *4th International Conference on
> Information Systems Security and Privacy (ICISSP)*, 2018.

The dataset is distributed by the University of New Brunswick under its own terms. Check the
landing page before redistributing the CSVs. This repository ships **no** dataset files, only the
code that reads them.

## 5. Provenance recorded with every result

For each run, `results/metrics.json` stores:

```json
"dataset": {
  "name": "CICIDS2017",
  "origin": "local_csv",
  "is_synthetic": false,
  "files": ["Monday-WorkingHours.pcap_ISCX.csv", "..."],
  "file_hashes_sha256": {"Monday-WorkingHours.pcap_ISCX.csv": "…"},
  "raw_rows": 529918,
  "rows_loaded": 200000,
  "label_values": {"BENIGN": 150000, "DoS Hulk": 30000, "...": 0},
  "sampling": {"max_rows_per_file": 25000, "strategy": "head", "raw_rows_on_disk": 529918},
  "notes": ["prototype subset: at most 25000 rows per file ..."],
  "retrieved_at_utc": "2026-10-14T09:00:00Z"
}
```

The SHA-256 hashes let a reviewer confirm that a reported run used a specific set of files.
