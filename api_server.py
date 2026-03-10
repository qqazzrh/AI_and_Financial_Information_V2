#!/usr/bin/env python3
"""FastAPI server for the biotech disclosure pipeline.

Uses the refactored Python pipeline package directly (no notebook execution)
and exposes async endpoints for ticker lookup, tiered pipeline execution
(with background polling), and cached result retrieval.

Usage:
    uvicorn api_server:app --reload --port 8000
    # or
    python3 api_server.py
"""

import asyncio
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from db import init_databases, get_user_history
from pipeline import (
    run_full_pipeline as _run_full_pipeline,
    run_tiered_pipeline as _run_tiered_pipeline,
    resolve_company_from_ticker as _resolve_company,
    set_database_handles as _set_database_handles,
    TieredPipelineRequest as _TieredPipelineRequest,
    PIPELINE_CONFIG as _PIPELINE_CONFIG,
    GLOBAL_CONFIG as _GLOBAL_CONFIG,
)

load_dotenv()

# ── Database initialization ─────────────────────────────────────────────────

_sqlite_conn, _lance_db = init_databases()
_set_database_handles(_sqlite_conn, _lance_db)

# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Biotech Disclosure Pipeline API",
    version="0.2.0",
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
        "version": "0.2.0",
        "worker_model_name": fmc.get("worker_model_name"),
        "embedding_provider": _GLOBAL_CONFIG.get("embedding_defaults", {}).get("provider"),
        "enable_embeddings": ff.get("enable_embeddings"),
        "databases": {
            "sqlite": _sqlite_conn is not None,
            "lancedb": _lance_db is not None,
        },
        "api_keys_configured": {
            "moonshot": bool(os.environ.get("MOONSHOT_API_KEY")),
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
    enable_graph_context: bool = False


def _run_pipeline_background(
    run_id: str,
    ticker: str,
    company_name: str | None,
    user_id: str,
    enable_graph_context: bool,
) -> None:
    """Execute the tiered pipeline synchronously (called from executor thread)."""
    RUN_STORE[run_id]["status"] = "running"
    try:
        request = _TieredPipelineRequest(
            ticker=ticker,
            company_name=company_name,
            user_id=user_id,
            enable_graph_context=enable_graph_context,
            run_id=run_id,
        )
        result = _run_tiered_pipeline(request)

        # Write payload to disk
        output_dir = os.path.dirname(__file__)
        fp = result["final_payload"]
        payload_json = fp.model_dump(mode="json")

        payload_path = os.path.join(output_dir, f"output_{ticker.upper()}_payload.json")
        with open(payload_path, "w") as f:
            json.dump(payload_json, f, indent=2, default=str)

        RUN_STORE[run_id]["status"] = "complete"
        RUN_STORE[run_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        RUN_STORE[run_id]["final_status"] = fp.status.value
        RUN_STORE[run_id]["analysis_tier"] = result["analysis_tier"]
        RUN_STORE[run_id]["cache_hits"] = result["cache_hits"]
        RUN_STORE[run_id]["cache_misses"] = result["cache_misses"]
    except Exception as exc:
        RUN_STORE[run_id]["status"] = "failed"
        RUN_STORE[run_id]["completed_at"] = datetime.now(timezone.utc).isoformat()
        RUN_STORE[run_id]["error"] = str(exc)


@app.post("/analyze/{ticker}")
async def analyze(ticker: str, request: Request, body: AnalyzeRequest | None = None):
    """Start a tiered pipeline run in the background. Returns a run_id for polling.

    Accepts X-User-ID header for user identity (defaults to 'anonymous').
    Set enable_graph_context=true in the body for Premium (vector+graph) execution.
    """
    user_id = request.headers.get("X-User-ID", "anonymous")
    company_name = body.company_name if body else None
    enable_graph_context = body.enable_graph_context if body else False
    analysis_tier = "vector_graph" if enable_graph_context else "text_only"

    run_id = str(uuid.uuid4())
    RUN_STORE[run_id] = {
        "run_id": run_id,
        "ticker": ticker.upper(),
        "company_name": company_name,
        "user_id": user_id,
        "analysis_tier": analysis_tier,
        "status": "pending",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "error": None,
    }
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _executor,
        _run_pipeline_background,
        run_id,
        ticker,
        company_name,
        user_id,
        enable_graph_context,
    )
    return {
        "run_id": run_id,
        "ticker": ticker.upper(),
        "analysis_tier": analysis_tier,
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


@app.get("/past_results")
def past_results(request: Request):
    """Retrieve query history for the authenticated user.

    User identity is read from the X-User-ID header (defaults to 'anonymous').
    Returns past queries with their analysis tier and arbiter responses.
    """
    user_id = request.headers.get("X-User-ID", "anonymous")
    history = get_user_history(_sqlite_conn, user_id)
    return {
        "user_id": user_id,
        "count": len(history),
        "queries": history,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
