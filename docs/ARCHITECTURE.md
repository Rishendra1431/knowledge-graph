# Architecture

## 1. Pipeline

```
data/raw/*.csv
      │
      ▼
[01] data acquisition ────────► dataset provenance (files, SHA-256, sampling, label census)
      │
      ▼
[02] preprocessing ───────────► cleaned frame + encoded features (everything fitted on TRAIN only)
      │                          ├─ timestamp parsing → day names
      │                          ├─ numeric coercion, inf → NaN
      │                          ├─ duplicate removal (before splitting)
      │                          ├─ corpus split (temporal / group / random)
      │                          └─ leakage audit (recorded, not silent)
      ▼
[03] threat-intel index ──────► per-IP malicious evidence derived from TRAIN LABELS ONLY
      │
      ▼
[04] knowledge graph ─────────► entities, relationships, provenance registry, confidence rules
      │                          + degree-ranked subgraph persisted for the dashboard
      ▼
[05] graph features ──────────► structural (label-free) + relational-supervised features
      │
      ▼
[06] detectors ───────────────► baseline_rf  (flow features)
      │                          graph_aware_rf (flow features + graph features)
      │                          same algorithm, hyperparameters, seed, split
      ▼
[07] explanations ────────────► per-alert: score, features, entities, paths, evidence, limitations
      │
      ▼
[08] statistics ──────────────► paired bootstrap + exact McNemar + rule-based claim
      │
      ▼
[09] resources ───────────────► wall time, CPU, peak RAM, tracemalloc, throughput, overhead
      │
      ▼
[10] validation & write ──────► consistency checks → results/*
      │
      ▼
scripts/verify_results.py ────► recomputes everything from saved predictions
```

Each numbered stage is wrapped in a `ResourceMonitor`, so its runtime, CPU and RAM appear in
`results/resources.csv` and `results/metrics.json`.

## 2. Module map

| Module | Responsibility |
| --- | --- |
| `src/config.py` | settings dataclasses, `config.yaml` loading, environment capture, logging |
| `src/data/acquire.py` | CSV reading/robustness, schema normalisation, download, synthetic fallback |
| `src/data/preprocess.py` | cleaning, target construction, splitting, encoding, leakage audit |
| `src/graph/builder.py` | entity/relationship construction, provenance registry, confidence rules, intel index |
| `src/graph/features.py` | graph-derived feature extraction + per-feature documentation |
| `src/graph/paths.py` | IP-level structural graph, BFS shortest paths to intel, path narration |
| `src/models/detectors.py` | `ThreatDetector` wrapper, baseline and graph-aware factories |
| `src/eval/metrics.py` | metrics, paired bootstrap, McNemar, claim rule |
| `src/eval/resources.py` | `ResourceMonitor`, `ResourceTracker`, overhead table |
| `src/eval/results.py` | result schema, consistency validation, CSV/JSON writers |
| `src/eval/runner.py` | orchestration of stages 01–10 |
| `src/explain/explainer.py` | alert explanations |
| `src/report/build_report.py` | fills `docs/RESULTS_TEMPLATE.md` from measured values |

## 3. Knowledge graph schema

### Nodes (created only when the data supports them)

| Type | ID pattern | Source | Attributes |
| --- | --- | --- | --- |
| `ip` | `ip:<addr>` | `Source IP`, `Destination IP` | `is_internal`, `evidence_count`, `first_seen`, `last_seen`, `roles`, `confidence` |
| `host` | `host:<addr>` | internal (RFC1918/loopback) IPs | `is_internal`, `evidence_count` |
| `service` | `svc:<name>/<port>` | `Destination Port` (+ well-known name lookup) | `port`, `service_name`, `evidence_count` |
| `protocol` | `proto:<n>` | `Protocol` | `protocol`, `evidence_count` |
| `event` | `evt:<record_id>` | one per flow record | `first_seen` |
| `technique` | `tech:<category>` | dataset labels (opt-in) | `note` = label-derived |
| `intel` | `intel:<addr>` | train labels only | `evidence_count`, `source` |

### Relationships

| Relation | Direction | Meaning |
| --- | --- | --- |
| `COMMUNICATES_WITH` | ip → ip | aggregated observed flows between two addresses |
| `USES_SERVICE` | ip → service, event → service | observed use of a port/service |
| `RUNS_ON_PROTOCOL` | service → protocol | observed port/protocol combination |
| `INITIATED_BY` | event → ip | the event's source address |
| `TARGETED` | event → ip | the event's destination address |
| `ALIAS_OF` | ip → host | internal address mapped to a host entity (1:1 here) |
| `MATCHES_TECHNIQUE` | event → technique | dataset label; **opt-in and never a model feature** |
| `FLAGGED_AS_INTEL` | ip → intel | IP appeared in ≥1 training-labelled malicious record |

### Confidence rules (documented, never invented per edge)

Every edge stores a `confidence` **and** the `confidence_rule` string that produced it:

| Rule name | Formula | Used for |
| --- | --- | --- |
| `observed` | `1.0` | facts stated directly by a dataset row (event endpoints, services, protocols) |
| `aggregated` | `1 − exp(−evidence_count / K)`, `K = graph.derived_confidence_k` (default 10) | aggregated host↔host, host↔service, service↔protocol edges |
| `intel_smoothed_ratio` | `(n_malicious + 1) / (n_total + 2)` (Laplace) | threat-intel evidence on an IP |

Properties: `aggregated` is monotone increasing in evidence and saturates at 1; `intel_smoothed_ratio`
never reaches 0 or 1, so an IP with no evidence sits at the 0.5 prior rather than being declared
clean.

### Provenance

Nodes and edges carry a `provenance_id` that resolves (via `KnowledgeGraph.provenance_of`) to:

```json
{
  "provenance_id": "prov_000123",
  "dataset": "CICIDS2017",
  "origin": "local_csv",
  "created_by": "build_knowledge_graph.communicates_with_edges",
  "files": ["Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"],
  "record_id_sample": [10231, 10232, 10237],
  "n_supporting_records": 418,
  "created_at_utc": "2026-10-14T09:12:44Z"
}
```

## 4. Graph feature scopes (the leakage question)

Graph features come in two flavours:

- **Structural, label-free.** Degree, fan-out/fan-in, common neighbours, Jaccard overlap,
  clustering, component and community membership, internal/external flags, port/service/protocol
  diversity, event shares. No label ever enters these.
- **Relational-supervised.** Counts and Laplace-smoothed ratios derived from the **training labels**
  through the threat-intel index, plus hop distances to intel-flagged IPs. Legitimate supervised
  features; every one is flagged `uses_training_labels: true` in `FEATURE_DESCRIPTIONS` and
  disclosed in any alert that uses it.

The *scope* of the structural graph is configurable because it changes what the experiment means:

| `graph.structural_scope` | Meaning | Risk |
| --- | --- | --- |
| `full` (default) | one graph over **all evaluated events** (train + test), labels never used | transductive: test-time topology influences test features |
| `train_only` | graph built from training events only; unseen entities fall back to sentinels | inductive and deployment-realistic, but weaker |

The runner evaluates the configured scope and then re-runs the graph-aware detector with
`train_only` as a **leakage-sensitivity experiment** (`results.metrics.json → leakage_sensitivity`).
If the inductive score collapses, part of the headline result came from test-time graph context.
Report both.

## 5. Splitting and leakage control

| Strategy | How | Why |
| --- | --- | --- |
| `temporal` (default) | train on configured weekdays, test on the rest (Mon–Thu / Fri by default) | mimics deployment: the model meets attack types not seen in training |
| `group` | `GroupShuffleSplit` on `(Source IP, Destination IP)` | stops one conversation appearing in both splits |
| `random` | stratified `train_test_split` | baseline comparison; weakest guarantee, used as the fallback |

Applied controls:

1. duplicates removed **before** splitting;
2. all cleaning statistics (medians, clip quantiles, vocabularies, dropped columns) fitted on train;
3. target, `Label`, `attack_category`, timestamps and raw IP/port identifiers never model features;
4. `leakage_audit` runs every execution and its verdict is stored — including a content-hash check
   for identical rows across splits and a chronology check for temporal splits;
5. a failed audit sets the pipeline status to `partial` and is surfaced in the report.

## 6. Detectors

Both arms are `RandomForestClassifier` with identical hyperparameters, `class_weight="balanced"`,
the same seed within a run and the same split. Seeds vary across repeated runs so metric variance is
visible, but each pair of arms shares its seed, keeping the comparison paired.

Batched inference (`evaluation.inference_batch_size`) bounds per-call memory and exposes per-batch
timing.

## 7. Statistics

- **Paired bootstrap** (`bootstrap_samples`, default 1000) over test records: both detectors are
  evaluated on the *same* resample, so the CI describes the difference rather than test difficulty.
- **Exact McNemar** test on discordant pairs for a distribution-free error-count comparison.
- **Claim rule** (`auto_claim`): a difference is asserted only when the bootstrap CI excludes zero;
  otherwise the result is reported as "no statistically detectable difference". The rule string is
  stored with the claim so a reviewer can check it.

## 8. Resource instrumentation

| Quantity | Source | Caveat recorded with the value |
| --- | --- | --- |
| Wall time | `time.perf_counter` | includes scheduling and I/O |
| CPU time | `psutil` `cpu_times` delta | most reliable CPU quantity |
| CPU percent | `psutil` sampler thread | relative to **one** core; >100% means several cores |
| Peak RAM | OS peak working set, else sampled RSS max | sampling can miss short spikes (lower bound) |
| Python allocations | `tracemalloc` peak | Python-level only; misses C-extension buffers; **not** total RAM |
| Throughput | `n_items / wall_seconds` | per-stage, per run |

One-time costs (training, graph construction) are separated from per-event costs (inference,
explanation) in the stage names and in `one_time_vs_per_event`.

## 9. Results schema and validation

`results/metrics.json` is versioned (`schema_version`). Structure:

```
status, measured, metrics_published, mode, generated_at_utc
environment      → OS, Python, CPU, RAM, package versions, config snapshot
dataset          → provenance block (see docs/DATASET.md)
preprocessing    → cleaning report, encoder metadata, feature lists
split            → strategy, sizes, class rates, per-side label distributions
leakage_audit    → per-check pass/fail + interpretation
leakage_sensitivity → inductive re-run of the graph-aware arm
threat_intel_index, knowledge_graph, graph_features, graph_scope
models           → per model: metrics, metrics per run, summary across runs,
                   predictor metadata, top features, inference timing
comparison       → bootstrap, McNemar, per-run deltas, rule-based claim
resources        → per-stage summary + every per-run measurement + caveats
resource_overhead→ baseline vs graph-aware inference table
explanations     → method, alert count, path/relationship statistics
stages, failed_stages, artifacts, notes, validation
```

`validate_results` enforces, among other things, that:

- **precision, recall, F1 and accuracy equal the values recomputed from the confusion matrix**
  (a hand-edited or invented metric block fails);
- the confusion matrix sums to `n_samples`;
- a model that did not run has `metrics: null` and `metrics_status: "not_executed"`;
- no metric is `NaN`/`Inf`;
- environment, dataset and leakage audit are present.

If validation fails, `metrics_published` is set to `false` and a note is added. A failed stage makes
the status `partial`; a run with no measured model is `failed`. There is no path in this codebase
that writes a metric it did not compute.
