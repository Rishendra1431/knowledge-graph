# An Explainable Knowledge Graph-Based Approach for Cyber Threat Detection

A research prototype that detects suspicious security events, represents the observed
cyber entities as a knowledge graph, and explains every alert in terms an analyst can check.

---

## ⚠️ Status: code complete, **not yet executed**

**No metrics in this repository have been measured yet.** The environment in which this code
was written had no working shell, so the pipeline has never been run here. Consequently:

- `results/metrics.json` does not exist until you run `python run_all.py`.
- The comparison table, resource numbers and confusion matrices are **empty templates**.
- Every placeholder in `docs/RESULTS_TEMPLATE.md` renders as `NOT MEASURED` until real data
  flows through it.

Run `python run_all.py` once and the artifacts fill in with genuinely measured values. If a
stage fails, the artifacts say so rather than showing a plausible-looking number. Please do not
quote any figure from this project until `scripts/verify_results.py --strict` exits with code 0
on **your** run.

---

## What this project does

| Requirement | Where it is implemented |
| --- | --- |
| Dataset handling, cleaning, encoding, leakage prevention | `src/data/` |
| Knowledge graph with provenance and confidence | `src/graph/builder.py` |
| Graph-derived features (structural + relational) | `src/graph/features.py`, `src/graph/paths.py` |
| Baseline and graph-aware detectors | `src/models/detectors.py` |
| Metrics, statistics, resource measurement | `src/eval/` |
| Explainable alerts | `src/explain/explainer.py` |
| Dashboard | `app/dashboard.py` |
| Paper-ready reporting | `src/report/build_report.py`, `docs/RESULTS_TEMPLATE.md` |
| Independent verification of every number | `scripts/verify_results.py` |

### The three claims the design protects

1. **No fabricated metrics.** Metrics are computed from arrays, validated for internal
   consistency (precision/recall/F1 are recomputed from the confusion matrix), and independently
   re-derived from saved predictions by `scripts/verify_results.py`.
2. **No hidden leakage.** Every cleaning statistic is fitted on the training split, duplicates are
   removed before splitting, a leakage audit runs on every execution, and a leakage-sensitivity
   experiment compares a transductive graph scope against a strict inductive one.
3. **No overstated explanations.** Feature attributions name their method, graph paths are
   labelled as proximity context, no explanation claims to prove malicious intent, and every
   alert states which of its features come from training-label hindsight.

---

## Project structure

```
kg-threat-detection/
├── run_all.py                     # one command: evaluate + verify
├── config.yaml                    # all tunable settings (code defaults exist too)
├── requirements.txt
├── conftest.py                    # pytest fixtures
├── README.md
├── app/
│   └── dashboard.py               # Streamlit dashboard (viewer only)
├── src/
│   ├── config.py                  # settings, environment capture, logging
│   ├── data/
│   │   ├── acquire.py             # CICIDS2017 loader, downloader, synthetic fallback
│   │   └── preprocess.py          # cleaning, encoding, splitting, leakage audit
│   ├── graph/
│   │   ├── builder.py             # knowledge graph, provenance, confidence rules
│   │   ├── features.py            # graph-derived features + documentation
│   │   └── paths.py               # IP graph, shortest paths to threat intel
│   ├── models/
│   │   └── detectors.py           # baseline and graph-aware RandomForest
│   ├── eval/
│   │   ├── metrics.py             # metrics, paired bootstrap, McNemar, claim rule
│   │   ├── resources.py           # psutil/tracemalloc/perf_counter instrumentation
│   │   ├── results.py             # schema, validation, artifact writers
│   │   └── runner.py              # the reproducible evaluation pipeline
│   ├── explain/
│   │   └── explainer.py           # per-alert explanations
│   └── report/
│       └── build_report.py        # fills the paper template
├── scripts/
│   └── verify_results.py          # recompute metrics from saved predictions
├── tests/                         # pytest suite (no real dataset required)
├── docs/
│   ├── DATASET.md                 # data source, features, labels, limitations
│   ├── ARCHITECTURE.md            # design, graph schema, confidence rules
│   ├── LIMITATIONS.md             # what this prototype cannot support
│   ├── SETUP_WINDOWS.md           # Windows + VS Code, step by step
│   └── RESULTS_TEMPLATE.md        # paper results template
├── data/                          # raw/ (you supply), interim/, processed/
├── results/                       # every measured artifact lands here
├── models/                        # trained model dumps (optional)
└── logs/                          # one log file per run
```

---

## Installation (Windows + VS Code)

Full walkthrough with troubleshooting: **[docs/SETUP_WINDOWS.md](docs/SETUP_WINDOWS.md)**.
The short version, in a VS Code terminal:

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # PowerShell/cmd;  source .venv/Scripts/activate in Git Bash

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3. Run the whole study (works with no dataset: a documented synthetic fallback is used)
python run_all.py

# 4. Open the dashboard
streamlit run app/dashboard.py
```

To use the real dataset, download the CICIDS2017 `MachineLearningCSV` files and put them in
`data/raw/` **before** running. See [docs/DATASET.md](docs/DATASET.md).

---

## Execution

```bash
python run_all.py                       # evaluate, then verify (default)
python run_all.py --n-runs 5            # more repeated runs for mean ± std
python run_all.py --max-rows 200000     # larger prototype subset
python run_all.py --max-rows 0          # all rows (long runtime, lots of RAM)
python run_all.py --verify-only         # re-verify existing artifacts
python scripts/verify_results.py --strict
streamlit run app/dashboard.py
pytest -q                               # unit tests, no dataset needed
```

Exit codes: `0` measured and verified · `1` verification failed · `2` not fully measured.

---

## What the pipeline produces

| Artifact | Content |
| --- | --- |
| `results/metrics.json` | every measured value, the environment fingerprint, validation verdict |
| `results/comparison_table.csv` | baseline vs graph-aware across all headline metrics |
| `results/metrics_headline.csv` | long format `(model, metric, value)` |
| `results/alerts.csv`, `results/alerts.json` | explained alerts with evidence and limitations |
| `results/resources.csv` | per-run CPU/RAM/runtime/throughput measurements |
| `results/feature_importance.csv` | Gini importances per detector |
| `results/leakage_audit.json` | the outcome of each leakage check |
| `results/knowledge_graph_subgraph.json` | degree-ranked subgraph for visualisation |
| `results/predictions/test_predictions.npz` | raw test predictions, for independent verification |
| `results/reports/results_for_paper.md` | the filled paper template |
| `results/reports/comparison_table.md`, `resource_evaluation.md` | standalone tables |

---

## Experiments

1. **Baseline vs graph-aware detection** — same algorithm, hyperparameters, seed and split; only
   the feature set differs (flow features vs flow features + graph features).
2. **Leakage sensitivity** — the graph-aware detector is re-run with the structural graph restricted
   to training events (inductive setting). A large drop means the transductive result relied on
   test-time graph context.
3. **Resource overhead** — runtime, CPU and peak RAM for training and per-event inference of both
   detectors, over repeated runs.
4. **Explanation audit** — every alert is checked for a traceable evidence source, a declared
   attribution method, and stated limitations.
5. **Repeated runs** — seeds vary across runs to expose fitting variability; both arms share the
   seed within a run so the comparison stays paired.

### Comparison table

Generated, never hand-written: `results/reports/comparison_table.md` and
`results/comparison_table.csv`. The verdict sentence is produced by a recorded rule — a difference
is claimed **only** if the paired-bootstrap confidence interval excludes zero.

---

## Reproducing a number

```bash
python run_all.py
python scripts/verify_results.py --strict
```

`verify_results.py` re-reads `results/predictions/test_predictions.npz`, recomputes every metric
from scratch, and fails if any published value disagrees. It also checks that each alert's threat
score equals the model output for that exact record.

---

## License and data

Code: use it freely for research. The CICIDS2017 dataset is distributed by the Canadian Institute
for Cybersecurity and its own terms apply — see [docs/DATASET.md](docs/DATASET.md) for provenance
and citation details. The bundled synthetic generator exists only so the pipeline is runnable
without the real data; it is labelled as synthetic everywhere.
