# Setup and execution on Windows + VS Code

This guide assumes Windows 10/11 and that you will run everything inside VS Code's integrated
terminal. Commands are shown for PowerShell/CMD and, where it differs, for Git Bash.

---

## 1. Install Python 3.10–3.12

1. Download from <https://www.python.org/downloads/windows/> (64-bit installer).
2. **Tick "Add python.exe to PATH"** on the first installer screen.
3. Verify in a new terminal:

```powershell
python --version
python -m pip --version
```

If `python` opens the Microsoft Store, install from python.org and reopen the terminal.

## 2. Install VS Code and the Python extension

1. Install VS Code from <https://code.visualstudio.com/>.
2. In VS Code, open the Extensions view (`Ctrl+Shift+X`) and install **Python** (Microsoft).
   Recommended extras: **Pylance**, **Jupyter**.

## 3. Open the project and create a virtual environment

```powershell
cd path\to\kg-threat-detection
code .                      # open the folder in VS Code
```

Then, in the VS Code terminal (`` Ctrl+` ``):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell
# .venv\Scripts\activate.bat      # CMD
# source .venv/Scripts/activate   # Git Bash
```

If PowerShell blocks activation with an execution-policy error, run this once and retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Select the interpreter in VS Code

`Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.\.venv\Scripts\python.exe`.
The status bar should now show `.venv`.

## 4. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- `shap` is optional. If it fails to build, skip it (`pip install -r requirements.txt --no-deps` is
  not needed — just install the rest and omit `shap`). Per-alert attributions then fall back to a
  documented approximation and say so in every alert.
- If a package fails to build, try a wheel-only install: `pip install --only-binary=:all: pandas numpy scikit-learn networkx`.

## 5. Put the dataset in place (optional but recommended)

1. Download the CICIDS2017 `MachineLearningCSV.zip` from
   <https://www.unb.ca/cic/datasets/ids-2017.html>.
2. Extract the CSVs into `data\raw\` (subfolders are fine):

```
data\raw\Monday-WorkingHours.pcap_ISCX.csv
data\raw\Tuesday-WorkingHours.pcap_ISCX.csv
...
```

The loader reads every `*.csv` under `data\raw\`. If the folder is empty, the pipeline runs on a
**documented synthetic dataset** instead and flags every artifact as synthetic — useful for a first
smoke test, never for reporting.

Edit `config.yaml` to change how much of each file is read:

```yaml
dataset:
  max_rows_per_file: 25000     # 0 = read everything (slow, lots of RAM)
  sample_strategy: head        # head | random
```

## 6. Run the study

```powershell
python run_all.py
```

This performs acquisition → preprocessing → graph build → both detectors → statistics →
explanations → resource measurement → validation → report, then verifies the artifacts.

Useful variants:

```powershell
python run_all.py --max-rows 50000 --n-runs 3     # small and quick
python run_all.py --max-rows 0                    # full dataset, expect long runtime
python run_all.py --max-alerts 50                 # explain more alerts
python run_all.py --no-verify                     # skip the verification step
python run_all.py --verify-only                   # re-verify existing results
```

Exit codes: `0` measured and verified · `1` verification failed · `2` not fully measured.

Then verify independently (this is the step to cite):

```powershell
python scripts/verify_results.py --strict
```

## 7. Open the dashboard

```powershell
streamlit run app/dashboard.py
```

A browser tab opens at <http://localhost:8501>. If the command is not found, use
`python -m streamlit run app/dashboard.py`.

The dashboard reads `results\metrics.json`. Until you run the pipeline it will show a clear "no
results" message — it never displays placeholder numbers.

## 8. Run the tests (no dataset needed)

```powershell
pip install pytest
pytest -q
```

The tests use a synthetic fixture and small RandomForests, so they complete in a couple of minutes.

---

## What you should see after a successful run

```
results\
├── metrics.json                     # every measured value + validation verdict
├── metrics_headline.csv             # long format (model, metric, value)
├── comparison_table.csv             # baseline vs graph-aware
├── alerts.csv / alerts.json         # explained alerts
├── resources.csv                    # per-run CPU/RAM/runtime/throughput
├── feature_importance.csv
├── leakage_audit.json
├── knowledge_graph_subgraph.json
├── predictions\test_predictions.npz
└── reports\
    ├── results_for_paper.md         # filled paper template
    ├── comparison_table.md
    └── resource_evaluation.md
```

Check `metrics.json` first: `status` should be `measured`, `metrics_published` should be `true`, and
`validation.valid` should be `true`. If `status` is `partial` or `failed`, look at `failed_stages`
and the log file in `logs\`.

---

## Troubleshooting

**`python` not recognised** — Python is not on PATH. Reinstall with "Add python.exe to PATH", or use
the full path `C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe`.

**Virtual environment will not activate** — see the execution-policy command in section 3, or run
`cmd` and use `.venv\Scripts\activate.bat`.

**`pip install` fails on `shap`** — omit `shap`. The explainer detects its absence and uses the
documented approximation, labelling it in every alert.

**`streamlit` not recognised** — use `python -m streamlit run app/dashboard.py`.

**Pipeline exits with code 2** — results were written but not fully measured. Open
`results\metrics.json`, read `failed_stages` and `notes`, and check the newest log in `logs\`.

**Memory error during training** — reduce the data and the forest size:

```yaml
dataset: { max_rows_per_file: 10000 }
model: { n_estimators: 100, n_jobs: 2 }
```

**Very long runtime** — reduce `evaluation.n_runs` to 1 while iterating, and lower
`evaluation.bootstrap_samples` to 300. Restore them for the final reported runs.

**The run is using the synthetic dataset by accident** — check `dataset.origin` in
`results\metrics.json`. If it is `synthetic`, `data\raw\` was empty or the CSV headers were not
recognised.

**Graph features are missing / graph-aware arm is `not_executed`** — the dataset has no usable
`Source IP` / `Destination IP` columns. See `docs/DATASET.md` section 3 for remapping the column
names in `config.yaml`.

**Timestamps fail to parse** — the split falls back to random and records the reason in
`split.fallback_reason`. Set `preprocess.timestamp_column` to the right column, or set
`preprocess.split_strategy: group` to avoid timestamps altogether.

---

## Recommended workflow while developing

1. `python run_all.py --max-rows 20000 --n-runs 1 --max-alerts 5` for a fast loop.
2. Inspect `results\metrics.json` and the dashboard.
3. When the setup is stable, run the final configuration (documented `n_runs`, full or intended
   subset) and archive the `results\` folder together with `config.yaml`.
4. Only quote numbers from an archived run whose `python scripts/verify_results.py --strict`
   exited with code 0.
