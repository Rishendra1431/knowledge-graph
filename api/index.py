"""Minimal Vercel entrypoint for the Python project.

The research dashboard remains a Streamlit application and is launched with
``streamlit run app/dashboard.py``. Vercel's Python runtime needs a callable
entrypoint, so this handler provides a small deployment health page instead of
trying to execute Streamlit inside a serverless function.
"""
from __future__ import annotations

import json
from typing import Iterable


def app(environ: dict, start_response) -> Iterable[bytes]:
    payload = {
        "name": "Explainable Knowledge Graph Threat Detection",
        "status": "ok",
        "dashboard": "Streamlit dashboard is available via app/dashboard.py",
    }
    body = json.dumps(payload).encode("utf-8")
    start_response(
        "200 OK",
        [
            ("Content-Type", "application/json; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]


handler = app
