"""One-command entry point: run the full evaluation, then verify it.

    python run_all.py                     # evaluate + verify
    python run_all.py --n-runs 5          # more repeated runs
    python run_all.py --max-rows 100000   # larger prototype subset
    python run_all.py --verify-only       # re-verify existing artifacts
    python run_all.py --no-verify         # evaluate only

Exit codes: 0 = measured and verified, 1 = verification failed, 2 = not fully measured.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings, setup_logging  # noqa: E402
from src.eval.runner import run_pipeline  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the explainable KG-based threat detection evaluation and verify the results."
    )
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--mode", default="full", help="label recorded in the results file")
    parser.add_argument("--n-runs", type=int, default=None, help="override evaluation.n_runs")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="override dataset.max_rows_per_file (0 = use every row)")
    parser.add_argument("--max-alerts", type=int, default=None, help="override explain.n_alerts")
    parser.add_argument("--seed", type=int, default=None, help="override the base random seed")
    parser.add_argument("--no-verify", action="store_true", help="skip verification")
    parser.add_argument("--verify-only", action="store_true", help="only verify existing results")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = load_settings(args.config)
    setup_logging(settings)

    if args.n_runs is not None:
        settings.evaluation.n_runs = max(int(args.n_runs), 1)
    if args.max_rows is not None:
        settings.dataset.max_rows_per_file = None if int(args.max_rows) <= 0 else int(args.max_rows)
    if args.max_alerts is not None:
        settings.explain.n_alerts = max(int(args.max_alerts), 1)
    if args.seed is not None:
        settings.seed = int(args.seed)

    if not args.verify_only:
        print("=" * 78)
        print("EXPLAINABLE KNOWLEDGE GRAPH-BASED CYBER THREAT DETECTION")
        print("=" * 78)
        print(f"raw data dir        : {(PROJECT_ROOT / settings.dataset.raw_dir).resolve()}")
        print(f"rows per file       : {settings.dataset.max_rows_per_file or 'all'}")
        print(f"split strategy      : {settings.preprocess.split_strategy}")
        print(f"graph scope         : {settings.graph.structural_scope}")
        print(f"repeated runs       : {settings.evaluation.n_runs}")
        print(f"results dir         : {settings.path('results_dir')}")
        print()
        try:
            payload, _ = run_pipeline(settings, mode=args.mode, write=True)
        except Exception as exc:
            print()
            print(f"PIPELINE FAILED: {type(exc).__name__}: {exc}")
            print("A failure record was written to results/metrics.json; stages that did not complete")
            print("contain no metrics. See the log in logs/ for the full traceback.")
            return 1

        print()
        print("-" * 78)
        print(f"status              : {payload['status']}")
        print(f"metrics published   : {payload['metrics_published']}")
        print(f"knowledge graph     : "
              f"{(payload.get('knowledge_graph') or {}).get('n_nodes')} nodes, "
              f"{(payload.get('knowledge_graph') or {}).get('n_edges')} edges")
        print(f"leakage audit       : "
              f"{'PASSED' if (payload.get('leakage_audit') or {}).get('passed_all') else 'FAILED'}")
        print(f"alerts explained    : {(payload.get('explanations') or {}).get('n_alerts')}")
        claim = ((payload.get("comparison") or {}).get("primary") or {}).get("claim") or {}
        if claim.get("claim"):
            print(f"comparison claim    : {claim['claim']}")
        print()
        print("Next steps:")
        print("  result files      : results/                (metrics.json, comparison_table.csv, alerts.csv)")
        print("  paper report      : results/reports/results_for_paper.md")
        print("  dashboard         : streamlit run app/dashboard.py")
        print("-" * 78)

    if args.no_verify:
        return 0

    from scripts.verify_results import main as verify_main

    print()
    print("Running independent verification of the written artifacts...")
    return verify_main([])


if __name__ == "__main__":
    sys.exit(main())
