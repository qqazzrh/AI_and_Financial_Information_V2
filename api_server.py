#!/usr/bin/env python3
"""FastAPI server for the biotech disclosure pipeline.

Loads the notebook namespace once at startup and exposes async endpoints
for ticker lookup, pipeline execution (with background polling), and
cached result retrieval.

Usage:
    uvicorn api_server:app --reload --port 8000
    # or
    python3 api_server.py
"""

import asyncio
import json
import logging
import os
import re
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# ── Notebook loader ──────────────────────────────────────────────────────────

NB_PATH = os.path.join(os.path.dirname(__file__), "notebooks", "01_biotech_disclosure_pipeline.ipynb")

# Lines matching this pattern in Cell 110 trigger the demo pipeline at import
# time — we strip them so the server starts without running a full analysis.
_DEMO_BLOCK_START = re.compile(r"^_demo_ticker\s*=")


def _load_notebook_namespace() -> dict[str, Any]:
    """Execute all code cells (0-110) into a shared namespace, skipping the
    auto-run demo block at the end of Cell 110."""
    with open(NB_PATH) as f:
        nb = json.load(f)

    sources: list[str] = []
    code_cell_count = 0
    for cell in nb["cells"][:111]:  # cells 0-110 inclusive
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        code_cell_count += 1

        # Strip the demo execution block from the last code cell
        # (starts at `_demo_ticker = ...` and runs to end of cell)
        lines = source.split("\n")
        cleaned_lines = []
        hit_demo = False
        for line in lines:
            if _DEMO_BLOCK_START.match(line):
                hit_demo = True
            if not hit_demo:
                cleaned_lines.append(line)
        if hit_demo:
            source = "\n".join(cleaned_lines)

        sources.append(source)

    ns: dict[str, Any] = {"__name__": "__main__", "__builtins__": __builtins__}

    # Silence stdout (cell-level print/json.dumps) and noisy loggers during exec
    _real_stdout = sys.stdout
    _devnull = open(os.devnull, "w")
    sys.stdout = _devnull
    _pipeline_logger = logging.getLogger("biotech_disclosure_pipeline")
    _saved_level = _pipeline_logger.level
    _pipeline_logger.setLevel(logging.WARNING)

    try:
        for i, code in enumerate(sources):
            try:
                exec(compile(code, f"<cell_{i}>", "exec"), ns)
            except Exception as e:
                print(f"  [cell {i}] Warning: {type(e).__name__}: {e}", file=sys.stderr)
    finally:
        sys.stdout = _real_stdout
        _devnull.close()
        _pipeline_logger.setLevel(_saved_level)

    print(f"Notebook loaded: {code_cell_count} code cells executed")
    return ns


print("Loading notebook namespace (this may take a moment)...")
NS = _load_notebook_namespace()

# Extract the functions we need
_run_full_pipeline = NS["run_full_pipeline"]
_resolve_company = NS["resolve_company_from_ticker"]
_PIPELINE_CONFIG = NS["PIPELINE_CONFIG"]
_GLOBAL_CONFIG = NS["GLOBAL_CONFIG"]

# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Biotech Disclosure Pipeline API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory run store (results persist to disk via the pipeline itself)
RUN_STORE: dict[str, dict[str, Any]] = {}

# ThreadPoolExecutor for running the synchronous pipeline
_executor = ThreadPoolExecutor(max_workers=2)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Config summary and key availability."""
    ff = _GLOBAL_CONFIG["feature_flags"]
    fmc = _GLOBAL_CONFIG["future_model_config"]
    return {
        "status": "ok",
        "analysis_client_mode": ff.get("analysis_client_mode"),
        "worker_model_name": fmc.get("worker_model_name"),
        "embedding_provider": _GLOBAL_CONFIG.get("embedding_defaults", {}).get("provider"),
        "enable_embeddings": ff.get("enable_embeddings"),
        "api_keys_configured": {
            "moonshot": bool(os.environ.get("MOONSHOT_API_KEY")),
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "openfda": bool(os.environ.get("OPENFDA_API_KEY")),
            "voyage": bool(os.environ.get("VOYAGE_AI_API_KEY")),
        },
    }


@app.get("/ticker/{ticker}")
def ticker_lookup(ticker: str):
    """Resolve a ticker symbol via SEC EDGAR."""
    company_name, aliases, cik = _resolve_company(ticker)
    if cik is None:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker.upper()}' not found in EDGAR company_tickers.json")
    return {
        "ticker": ticker.upper(),
        "company_name": company_name,
        "cik": cik,
        "aliases": aliases,
        "resolved": True,
    }


class AnalyzeRequest(BaseModel):
    company_name: str | None = None


def _run_pipeline_background(run_id: str, ticker: str, company_name: str | None = None) -> None:
    """Execute the pipeline synchronously (called from executor thread)."""
    RUN_STORE[run_id]["status"] = "running"
    try:
        kwargs: dict[str, Any] = {"ticker": ticker}
        if company_name:
            kwargs["company_name"] = company_name
        result = _run_full_pipeline(**kwargs)

        # Write payload to disk (same as run_pipeline.py)
        output_dir = os.path.dirname(__file__)
        fp = result["final_payload"]
        payload_json = fp.model_dump(mode="json")

        payload_path = os.path.join(output_dir, f"output_{ticker.upper()}_payload.json")
        with open(payload_path, "w") as f:
            json.dump(payload_json, f, indent=2, default=str)

        RUN_STORE[run_id]["status"] = "complete"
        RUN_STORE[run_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        RUN_STORE[run_id]["final_status"] = fp.status.value
    except Exception as exc:
        RUN_STORE[run_id]["status"] = "failed"
        RUN_STORE[run_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        RUN_STORE[run_id]["error"] = str(exc)


@app.post("/analyze/{ticker}")
async def analyze(ticker: str, body: AnalyzeRequest | None = None):
    """Start a pipeline run in the background. Returns a run_id for polling."""
    company_name = body.company_name if body else None
    run_id = str(uuid.uuid4())
    RUN_STORE[run_id] = {
        "run_id": run_id,
        "ticker": ticker.upper(),
        "company_name": company_name,
        "status": "pending",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "error": None,
    }
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _run_pipeline_background, run_id, ticker, company_name)
    return {
        "run_id": run_id,
        "ticker": ticker.upper(),
        "status": "pending",
    }


@app.get("/status/{run_id}")
def status(run_id: str):
    """Poll the status of a pipeline run."""
    if run_id not in RUN_STORE:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return RUN_STORE[run_id]


@app.get("/results/{ticker}")
def results(ticker: str):
    """Read the cached FinalUIPayload JSON from disk."""
    payload_path = os.path.join(os.path.dirname(__file__), f"output_{ticker.upper()}_payload.json")
    if not os.path.exists(payload_path):
        raise HTTPException(status_code=404, detail=f"No cached results for ticker '{ticker.upper()}'")
    with open(payload_path) as f:
        return json.load(f)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
