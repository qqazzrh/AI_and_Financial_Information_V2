# PENRS System Readout
**Last Updated:** 2026-03-04
**Consolidation Phase:** Post Step 9 (Phase 4 — Clean Up complete)
**Architecture:** 3-notebook + 1 shared utility module

---

## What Is PENRS?

PENRS is a multi-agent financial intelligence pipeline for analyzing publicly traded biotech/pharma companies. It fetches and scores nine types of financial and scientific documents, runs them through specialized worker agents, aggregates scores via an arbiter, and synthesizes a weighted composite rating with narrative contradiction detection. The result is a "PENRS score" (−1.0 to +1.0) indicating bearish-to-bullish signal strength, saved as a structured JSON report on disk.

**Architecture summary:** Data Layer → Worker Agents → Arbiter → Master → JSON Report on disk.

**Current state:** The complete infrastructure layer is implemented and tested. Real API fetchers and real LLM calls are the remaining work — the plumbing is done, the water isn't connected yet.

---

## Repository File Inventory (Post Step 9)

```
AI_and_Financial_Information/
├── penrs_mcp_server.py      # FastMCP server entry point (standalone, never migrated)
├── utils.py                 # Consolidated utility module: Cache + HTTP + Rate Limit + Router
├── worker_nodes.ipynb       # PENRSWorker base class, truncation, JSON parsing
├── orchestrator.ipynb       # ArbiterAgent, MasterAgent, PENRSReport, run_penrs()
├── tests.ipynb              # Full test suite (38 test functions, 8 sections)
├── implementation_plan.md   # Sequential numbered step checklist with status markers
├── specs.md                 # Architecture and original product specifications
├── PENRS_SYSTEM_READOUT.md  # This file
├── .env.example             # Documented environment variable template
├── pytest.ini               # Pytest config (addopts = -v --tb=short)
├── prompt.md                # System prompt / original project brief
├── run_loop.sh              # Shell: run pipeline in a loop
├── run_loop_codex.sh        # Shell: codex variant of above
├── Test_files/              # Ephemeral test artifacts (temp dirs created by cache/pipeline tests)
└── .penrs_cache/            # Cache directory (auto-created on utils.py import)
```

**Deleted in Step 9** (logic now lives in notebooks / utils.py):
- Source: `penrs_worker.py`, `penrs_arbiter.py`, `penrs_pipeline.py`, `penrs_cache.py`, `penrs_http.py`, `penrs_rate_limit.py`, `penrs_router.py`
- Tests: `test_arbiter.py`, `test_cache.py`, `test_pipeline.py`, `test_rate_limit.py`, `test_router.py`, `test_step1.py`, `test_step2.py`, `test_worker.py`

---

## Detailed Component Reference

---

### 1. `penrs_mcp_server.py` — FastMCP Server Entry Point

**Status:** Unchanged. Standalone file, not absorbed into any notebook.
**Depends on:** `python-dotenv`, `fastmcp` (third-party only — zero project imports)

Initializes the FastMCP server that tool functions will eventually register onto. On startup:

1. Calls `load_dotenv()` to load `.env` from working directory
2. Reads three env vars with fallbacks:
   - `ALPHA_VANTAGE_API_KEY` → `"demo"`
   - `PENRS_CACHE_DIR` → `".penrs_cache"`
   - `PENRS_LOG_DIR` → `".penrs_logs"`
3. Calls `.mkdir(parents=True, exist_ok=True)` on both directories — they are always created at startup
4. Configures `logging.basicConfig` with:
   - Level: `INFO`
   - Format: `"%(asctime)s [%(levelname)s] %(name)s: %(message)s"`
   - Two handlers: `FileHandler(PENRS_LOG_DIR / "penrs.log")` + `StreamHandler()`
5. Creates `logger = logging.getLogger("penrs_mcp")` — this is the same logger name used throughout all project modules
6. Instantiates `mcp = FastMCP("penrs_mcp")`
7. Runs `mcp.run(transport="stdio")` when executed as `__main__`

**Usage:** `python penrs_mcp_server.py`

**Key design note:** This file intentionally imports nothing from the project. Future MCP tool functions will be added here as `@mcp.tool()` decorated functions. The server name `"penrs_mcp"` must match exactly — it is hardcoded into tests.

---

### 2. `utils.py` — Consolidated Utility Module

**Status:** Fully rewritten in Step 9. Previously a re-export shim; now contains all source code inline.
**Depends on:** `httpx` (third-party), stdlib only.

Single import point for all three notebooks. Organized into four sections under `## ─── Section ───` headers:

#### Section A: `## ─── Cache ───`

**Module-level initialization:**
- `PENRS_CACHE_DIR = Path(os.getenv("PENRS_CACHE_DIR", ".penrs_cache")).resolve()` — resolved to absolute path. This is evaluated at import time (and re-evaluated on `importlib.reload(utils)`).
- `PENRS_CACHE_DIR.mkdir(parents=True, exist_ok=True)` — auto-creates cache dir on import.
- `_META_FIELDS = {"_cached_at", "_api", "_ticker", "_doc_type", "_date"}` — set of keys that are metadata, not payload, used in the backward-compatibility fallback.

**`cache_key(api, ticker, doc_type, date=None) → str`**
- Builds the string `"{api}|{ticker}|{doc_type}|{date or ''}"` and SHA-256 hashes it
- Returns a 64-character lowercase hex digest
- Deterministic: same inputs always produce the same key
- Logs at INFO level on every call

**`_cache_path(api, ticker, doc_type, date=None) → Path`** (private)
- Calls `cache_key()` and returns `PENRS_CACHE_DIR / f"{key}.json"`

**`cache_set(api, ticker, doc_type, date, payload) → Path`**
- Builds a record dict:
  ```json
  {
    "_cached_at": "<ISO 8601 UTC>",
    "_api": "...", "_ticker": "...", "_doc_type": "...", "_date": "...|null",
    "payload": { <the actual data> }
  }
  ```
- Serializes with `json.dumps(..., ensure_ascii=True)` and writes to the cache path
- Returns the `Path` object of the written file

**`cache_get(api, ticker, doc_type, date, max_age_hours) → dict | None`**
- Returns `None` on: missing file, OSError, JSONDecodeError, missing `_cached_at`, unparseable `_cached_at`, or age ≥ `max_age_hours`
- Returns `payload` dict (the nested `"payload"` key) on hit
- **Backward-compat fallback:** if `"payload"` key is absent (legacy cache files), returns all non-metadata keys as the payload
- All misses log at INFO or WARNING; hits log at INFO

#### Section B: `## ─── HTTP ───`

**Module-level constants:**
- `_RETRY_STATUSES = {429, 503}` — HTTP status codes that trigger a retry
- `_MAX_RETRIES = 3` — maximum retry attempts after the initial request
- `_DEFAULT_TIMEOUT = 30.0` — seconds before an `httpx.TimeoutException`

**`_api_request(url, params=None, headers=None, api_name="unknown", timeout=30.0) → dict`** (async)
- Opens one `httpx.AsyncClient` for the entire request lifecycle (including all retries)
- Loops `_MAX_RETRIES + 1` times (1 initial attempt + up to 3 retries):
  1. Catches `httpx.TimeoutException` → returns `{"error": "Request timed out", "detail": f"URL: {url}"}`
  2. Catches `httpx.RequestError` (DNS failure, connection refused, etc.) → returns `{"error": "Request failed", "detail": str(exc)}`
  3. If status is in `_RETRY_STATUSES` AND still within retry budget: computes `wait = 2**(attempt-1)` (1s, 2s, 4s) and calls `await asyncio.sleep(wait)`, then continues loop
  4. If `response.is_error` (any non-retried 4xx/5xx): returns `{"error": "HTTP {code}", "detail": response.text[:500]}`
  5. On success: attempts `response.json()` — returns parsed dict, or `{"text": response.text}` if JSON parse fails
- After loop exhaustion (all retries failed with retry-eligible status): returns `{"error": "Max retries exceeded", "detail": f"URL: {url}"}`
- `api_name` appears in all log messages for tracing which API generated an error

#### Section C: `## ─── Rate Limit ───`

**Module-level constants:**
- `_ALPHA_DAILY_LIMIT = 25` — Alpha Vantage free tier daily cap
- `_ALPHA_MINUTE_LIMIT = 5` — Alpha Vantage free tier per-minute cap
- `_ALPHA_SLEEP_SECONDS = 12` — sleep duration when Alpha Vantage minute limit is hit
- `_SEC_MINUTE_LIMIT = 10` — SEC EDGAR per-minute cap
- `_DEFAULT_RPM_LIMIT = int(os.getenv("PENRS_DEFAULT_RPM_LIMIT", "60"))` — for all other APIs

**Module-level state:**
- `_RATE_LIMIT_STATE: dict[str, dict[str, Any]] = {}` — keyed by normalized API name; each value has `day_key`, `daily_count`, `minute_key`, `minute_count`
- `_RATE_LIMIT_LOCK = threading.Lock()` — guards all reads/writes to `_RATE_LIMIT_STATE`

**Private helpers:**
- `_now_utc()` — returns `datetime.now(timezone.utc)`; extracted for monkeypatching in tests
- `_normalize_api_name(api_name)` — strips, lowercases, replaces spaces and hyphens with underscores
- `_limits_for_api(api_name, rpm_limit)` → `(minute_limit, daily_limit)` — dispatches to hardcoded limits for known APIs; falls back to `(rpm_limit or _DEFAULT_RPM_LIMIT, None)`
- `_warn_if_approaching_or_hit(*, api_name, window, count, limit, blocked=False)` — emits WARNING log when `count == limit - 1` (approaching) or `count == limit` (hit); blocked=True forces the "reached" message unconditionally

**`_reset_rate_limit_state() → None`**
- Acquires lock, calls `_RATE_LIMIT_STATE.clear()`
- Used by tests to reset between runs; callable from `utils` module object

**`_check_rate_limit(api_name, rpm_limit=None) → bool`**
- Full flow under the lock:
  1. Normalizes api_name, resolves limits
  2. Gets or creates the per-API state dict with `setdefault`
  3. Rolls over `day_key`/`minute_key` if the current UTC time has crossed the boundary, resetting their counters to 0
  4. Checks daily limit: if `daily_count >= daily_limit` → logs WARNING, returns `False`
  5. Checks minute limit:
     - For Alpha Vantage: sleeps `_ALPHA_SLEEP_SECONDS` (12s), resets `minute_key` and `minute_count`, then **continues** (returns `True`)
     - For all others: logs WARNING, returns `False`
  6. Increments both `minute_count` and `daily_count`
  7. Emits approaching/hit warnings for both windows
  8. Returns `True`

**API limit table:**
| Normalized Name | Per-Minute | Per-Day | At Minute Limit | At Daily Limit |
|---|---|---|---|---|
| `alpha`, `alpha_vantage`, `alphavantage` | 5 | 25 | Sleep 12s, reset, continue | Return False |
| `sec`, `sec_edgar`, `edgar` | 10 | None | Return False | N/A |
| anything else | `rpm_limit` or 60 | None | Return False | N/A |

#### Section D: `## ─── Router ───`

**`DocumentType(str, Enum)`** — exactly 9 values:
```
EARNINGS_CALL     = "earnings_call"
FORM_4            = "form_4"
NEWS_SENTIMENT    = "news_sentiment"
PRICE_HISTORY     = "price_history"
SEC_10K           = "sec_10k"
SEC_10Q           = "sec_10q"
SEC_8K            = "sec_8k"
CLINICAL_TRIALS   = "clinical_trials"
BIOMEDICAL_EVIDENCE = "biomedical_evidence"
```
Inherits from both `str` and `Enum` so members compare equal to their string values.

**Type aliases:**
- `DateRange = tuple[str | None, str | None] | dict[str, str | None] | None`
- `Fetcher = Callable[[str, tuple[str|None, str|None], DocumentType], Awaitable[Any]]`

**`DOCUMENT_API_ROUTING: dict[DocumentType, tuple[str, ...]]`**
```
EARNINGS_CALL     → ("alpha_vantage",)
FORM_4            → ("alpha_vantage",)
NEWS_SENTIMENT    → ("alpha_vantage",)
PRICE_HISTORY     → ("alpha_vantage",)
SEC_10K           → ("sec_edgar",)
SEC_10Q           → ("sec_edgar",)
SEC_8K            → ("sec_edgar",)
CLINICAL_TRIALS   → ("clinicaltrials_gov",)
BIOMEDICAL_EVIDENCE → ("openfda", "pubmed")   ← multi-source, both run concurrently
```

**Private helpers:**
- `_normalize_date_range(date_range)` — accepts `None`, 2-tuple, or `{"from": ..., "to": ...}` dict; always returns a `(str|None, str|None)` tuple; raises `ValueError`/`TypeError` on bad input
- `_missing_fetcher(api_name, ...)` — async stub that returns `{"error": "No fetcher configured for API '...'"}` when no real fetcher is registered for an API name
- `_is_usable_data(payload)` — returns False for None, empty string, empty collection; True otherwise
- `_extract_source_data(result)` — unwraps the status envelope: returns `result["data"]` for `status=="available"`, `None` for `status=="not_released"` or `"error"` key, the raw result otherwise

**`penrs_fetch_document(ticker, document_type, date_range=None, fetchers=None) → dict`** (async)
- Raises `TypeError` if `document_type` is not an actual `DocumentType` instance (string rejected)
- Normalizes `date_range` to `(str|None, str|None)` tuple
- Looks up `DOCUMENT_API_ROUTING[document_type]` to get the list of APIs
- For each API: uses the registered `fetchers[api]` coroutine, or `_missing_fetcher` if none registered
- Runs all API coroutines concurrently via `asyncio.gather(..., return_exceptions=True)`
- Classifies results into `sources` (usable) and `partial_failures` (errors/empty)
- Return envelope:
  - At least one source succeeded: `{"status": "available", "data": {"ticker", "document_type", "date_range", "apis_attempted", "sources": [...], "partial_failures": [...]}}`
  - All failed: `{"status": "not_released", "data": {"ticker", ..., "errors": [...]}}`

**`__all__`** lists: `PENRS_CACHE_DIR`, `cache_get`, `cache_key`, `cache_set`, `_api_request`, `_check_rate_limit`, `_reset_rate_limit_state`, `DOCUMENT_API_ROUTING`, `DocumentType`, `penrs_fetch_document`.

---

### 3. `worker_nodes.ipynb` — Worker Base Class

**Migrated from:** `penrs_worker.py`
**Depends on:** `utils.py` (`DocumentType`, `penrs_fetch_document`), stdlib only.
**External deps:** none beyond utils.

#### Cell 1 — Markdown title
`# Worker Nodes` + 3-sentence summary of the notebook's purpose.

#### Cell 2 — Imports
```python
from __future__ import annotations
import json, re
from pathlib import Path
from typing import Any, Awaitable, Callable
from utils import DocumentType, penrs_fetch_document
```
Note: `from __future__ import annotations` enables Python 3.10+ style annotations (`X | Y`) on Python 3.9.

#### Cell 3 — Markdown: `## Type Aliases & Constants`

#### Cell 4 — Type aliases and constants
```python
RubricFetcher   = Callable[[str], Any | Awaitable[Any]]
DocumentFetcher = Callable[[str, DocumentType, dict[str, str] | None], Any | Awaitable[Any]]
LLMInvoker      = Callable[[str], Any | Awaitable[Any]]
_DEFAULT_RUBRICS_PATH      = Path("rubrics.json")
_TRUNCATION_MARKER_TEMPLATE = "\n...[truncated {count} chars]...\n"
```

#### Cell 5 — Markdown: `## Helper Utilities`

#### Cell 6 — Private helpers
- `_maybe_await(value)` — if `value` has `__await__`, awaits it; otherwise returns it directly. Allows injected callbacks to be either sync or async.
- `_load_rubric_from_json(rubric_id)` — reads `rubrics.json` from disk, returns the rubric dict for `rubric_id`. Returns an error dict (not raises) if file is missing, unreadable, or rubric not found.
- `_coerce_text(document_payload)` — converts any type to a string: strings pass through, bytes are decoded UTF-8, None becomes `""`, dicts/lists become JSON strings, everything else becomes `str(x)`.
- `_extract_json_from_text(text)` — scans for `{` or `[` characters, attempts `json.JSONDecoder.raw_decode()` at each one, returns the first successfully parsed object or `None`.

#### Cell 7 — Markdown: `## Truncation`

#### Cell 8 — `truncate_for_context(text, max_chars=12000) → str`
Splits the budget evenly between head and tail with a marker in the middle:
1. If `len(text) <= max_chars`: returns text unchanged
2. `truncated_count = len(text) - max_chars`
3. `marker = "\n...[truncated N chars]...\n"` where N is `truncated_count`
4. If marker itself is longer than `max_chars`: returns `text[:max_chars]` (degenerate case)
5. Otherwise: `remaining = max_chars - len(marker)`, `head_len = remaining // 2`, `tail_len = remaining - head_len`
6. Returns `f"{text[:head_len]}{marker}{text[-tail_len:]}"`

Result is always exactly `max_chars` characters. LLM sees both the beginning and the end of the document.

#### Cell 9 — Markdown: `## PENRSWorker Class`

#### Cell 10 — `class PENRSWorker`

**Constructor (`__init__`):**
| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `name` | str | required | Worker identifier string |
| `weight` | float | required | Relative scoring importance |
| `signal_density` | float | required | Drives star rating (0–1) |
| `rubric_id` | str | required | Key into `rubrics.json` |
| `document_type` | DocumentType | required | Determines which API to call |
| `rubric_fetcher` | RubricFetcher | `_load_rubric_from_json` | Injectable |
| `document_fetcher` | DocumentFetcher | `penrs_fetch_document` | Injectable |
| `llm_invoker` | LLMInvoker | `lambda _: "{}"` | Injectable (stub by default) |
| `max_context_chars` | int | 12000 | Truncation limit |

**`metadata` property:** returns `{"name": ..., "weight": ..., "signal_density": ...}`

**`parse_json_response(response) → dict`** — never raises:
1. If already a dict: return as-is
2. If list: return `{"items": response}`
3. If None or empty string: return `{"parse_error": "empty_response", ...}`
4. Try `json.loads(text)`
5. Try extracting from fenced ` ```json ... ``` ` block
6. Try `_extract_json_from_text` (regex scan for `{` or `[`)
7. Fallback: `{"parse_error": "unable_to_parse_json", "raw_response": text}`

**`build_prompt(*, ticker, date_from, date_to, rubric, document_excerpt) → str`:**
Returns a structured string:
```
Worker: {name}
Ticker: {ticker}
Date range: {date_from} -> {date_to}
Rubric JSON: {json.dumps(rubric)}
Document excerpt:
{document_excerpt}
Return only JSON.
```

**`run(ticker, date_from, date_to) → dict`** (async) — main execution:
1. `rubric = await _maybe_await(rubric_fetcher(rubric_id))` — fetches rubric, coerces to dict
2. `doc_result = await _maybe_await(document_fetcher(ticker, document_type, {"from": ..., "to": ...}))` — fetches document
3. If `doc_result["status"] == "not_released"`: returns early with `{"status": "not_released", "worker": metadata, "ticker", "date_from", "date_to", "document_type", "apis_attempted", "detail"}`
4. Extracts `doc_result["data"]`, coerces to string, truncates
5. Builds prompt, invokes LLM, parses response
6. Returns `{"status": "available", "worker": metadata, "ticker", "date_from", "date_to", "document_type", "rubric", "result": parsed_llm_output}`

---

### 4. `orchestrator.ipynb` — Orchestration & Pipeline

**Migrated from:** `penrs_arbiter.py`, `penrs_pipeline.py`
**Depends on:** `pydantic` (`BaseModel`, `ConfigDict`, `Field`), stdlib only.
**External deps:** `pydantic` must be installed.

#### Cell 1 — Markdown title
`# Orchestrator` + summary.

#### Cell 2 — Imports
```python
from __future__ import annotations
import asyncio, json, re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from pydantic import BaseModel, ConfigDict, Field
```

#### Cell 3 — Markdown: `## Arbiter Helpers`

#### Cell 4 — Arbiter constants and pure helpers
**`ARBITER_SYSTEM_PROMPT`** — string literal:
> "You are the Lead Portfolio Manager reviewing all worker outputs for consistency. You must flag mandatory contradictions when present: - Lipstick on a Pig - Bailing Out - Dilute and Delay. Return strictly valid JSON."

**`_MANDATORY_CONTRADICTIONS`** — tuple of 3 dicts, each with `name`, `severity`, `patterns`:
| Name | Severity | Regex Patterns |
|---|---|---|
| Lipstick on a Pig | High | `\blipstick on a pig\b` |
| Bailing Out | High | `\bbailing out\b`, `\bbail[- ]?out\b` |
| Dilute and Delay | Medium | `\bdilute and delay\b` |

Private helpers: `_require_field`, `_coerce_float`, `_normalize_score` (clamps to [-1, 1]), `_derive_star_rating` (thresholds: 0.85→5★, 0.65→4★, 0.45→3★, 0.25→2★, else 1★), `_extract_star_rating` (uses provided `star_rating` from result or derives from signal_density), `_collect_narrative_text` (concatenates summary/thesis/narrative/analysis fields), `_detect_mandatory_contradictions` (runs all 3 regex rules, returns 3-item list always).

#### Cell 5 — Markdown: `## ArbiterAgent`

#### Cell 6 — `class ArbiterAgent`
**`__init__(system_prompt=ARBITER_SYSTEM_PROMPT)`**

**`_validate_worker_result(worker_result)`** (private) — raises `ValueError` if any required field is missing: `status`, `worker` (dict), `result` (dict), `worker.name`, `worker.weight`, `worker.signal_density`, `result.score`.

**`evaluate(worker_results: list[dict]) → dict`:**
1. Iterates workers; skips any with `status != "available"`
2. For each available worker: validates schema, normalizes score, derives star rating, computes `effective_weight = base_weight × (star_rating / 5.0)`, computes `weighted_score = normalized_score × effective_weight`
3. Final `weighted_score = weighted_score_sum / total_effective_weight`, clamped to [-1, 1]
4. Detects contradictions from all concatenated narrative text
5. Returns:
```json
{
  "status": "available",
  "arbiter_role": "Lead Portfolio Manager",
  "system_prompt": "...",
  "worker_scores": [
    {"name", "raw_score", "normalized_score", "weight",
     "star_rating", "effective_weight", "weighted_score"}
  ],
  "weighted_score": 0.0,
  "contradictions": [
    {"name", "severity", "flagged": bool, "evidence": "str|null"}
  ]
}
```

#### Cell 7 — Markdown: `## Pipeline Helpers`

#### Cell 8 — Pipeline private helpers and `run_all_workers`
- `_maybe_await(value)` — same async-compat shim as in worker_nodes (duplicated intentionally, noted in specs)
- `_coerce_float_or_zero(value)` — like `_coerce_float` but returns 0.0 instead of raising
- `_worker_identity(worker)` — extracts `name`, `weight`, `signal_density` from a worker object via `getattr`
- `_invoke_worker(worker, ticker, date_from, date_to)` — finds `worker.run`, calls it, wraps with `_maybe_await`
- **`run_all_workers(workers, ticker, date_from, date_to) → list[dict]`** (async):
  - Empty input → returns `[]`
  - Creates coroutines list with `_invoke_worker` for each worker
  - `await asyncio.gather(*coroutines, return_exceptions=True)` — exception in one worker does NOT cancel others
  - For each result: if `Exception` → wraps as `{"status": "error", "worker": identity, "error": str(exc)}`; if non-dict → error; otherwise passes through
- `_build_arbiter_input(worker_results)` — filters to only `status=="available"` results with valid `worker` and `result` dicts
- `_evaluate_with_arbiter(arbiter, worker_results)` — calls `_build_arbiter_input`; if empty returns `{"status": "not_available", "weighted_score": 0.0, ...}`; otherwise calls `arbiter.evaluate()`; wraps exceptions as error dict
- `_safe_filename_component(value)` — replaces non-alphanumeric, non-hyphen, non-underscore chars with `_`; strips leading/trailing underscores; returns `"unknown"` if empty

#### Cell 9 — Markdown: `## MasterAgent`

#### Cell 10 — `class MasterAgent`
**`synthesize(*, ticker, date_from, date_to, worker_results, arbiter_result) → dict`** (sync method, not async):
- Counts workers with `status=="available"`
- Extracts `arbiter_result["weighted_score"]`
- Returns:
```json
{
  "status": "available",
  "model": "programmatic_master_v1",
  "ticker": "...",
  "date_range": {"from": "...", "to": "..."},
  "final_score": 0.0,
  "available_worker_count": 0,
  "total_worker_count": 0
}
```

#### Cell 11 — Markdown: `## PENRSReport Schema`

#### Cell 12 — `class PENRSReport(BaseModel)`
```python
model_config = ConfigDict(extra="allow")
ticker: str
date_from: str
date_to: str
generated_at: str
worker_results: list[dict[str, Any]] = Field(default_factory=list)
arbiter: dict[str, Any]
master: dict[str, Any]
report_path: str
```
`extra="allow"` means fields not listed above are silently accepted — future schema additions won't cause `ValidationError`.

#### Cell 13 — Markdown: `## run_penrs`

#### Cell 14 — `run_penrs(ticker, date_from, date_to, *, workers, arbiter, master, report_dir, now) → dict` (async)

The single top-level entry point for a full pipeline run:

1. `worker_results = await run_all_workers(workers or [], ticker, date_from, date_to)`
2. `arbiter_agent = arbiter or ArbiterAgent()` then `arbiter_result = _evaluate_with_arbiter(arbiter_agent, worker_results)`
3. `master_agent = master or MasterAgent()` then `master_result = await _maybe_await(master_agent.synthesize(...))`
4. Generates ISO 8601 UTC timestamp for `generated_at`
5. Creates `report_dir` with `mkdir(parents=True, exist_ok=True)`
6. Filename: `{safe_ticker}_{safe_date_from}_{safe_date_to}_{YYYYMMDDTHHMMSSz}.json`
7. Builds `report_payload` dict, validates via `PENRSReport.model_validate(report_payload)`, calls `.model_dump(mode="json")`
8. Writes to `report_path` as indented JSON (`indent=2, ensure_ascii=True`)
9. Returns the complete serialized report dict including `"report_path"` key

**Parameters with defaults:**
- `workers=None` → treated as empty list
- `arbiter=None` → `ArbiterAgent()` with default system prompt
- `master=None` → `MasterAgent()`
- `report_dir="penrs_reports"` — relative or absolute path
- `now=None` → `datetime.now(timezone.utc)`

---

### 5. `tests.ipynb` — Full Test Suite

**Migrated from:** `test_arbiter.py`, `test_cache.py`, `test_pipeline.py`, `test_rate_limit.py`, `test_router.py`, `test_step1.py`, `test_step2.py`, `test_worker.py`
**Depends on:** `utils.py`, `worker_nodes.ipynb` (via `%run`), `orchestrator.ipynb` (via `%run`), `penrs_mcp_server.py`

The notebook has 20 cells: 9 markdown section headers + 10 code cells + 1 runner cell.

#### Cell 2 — Imports (the critical cell)
```python
# Standard library
import asyncio, importlib, json, os, uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from unittest.mock import patch

# Third-party
import httpx
import pytest

# Utility module
import utils

# Aliases for monkeypatching tests:
#   test helpers reference penrs_cache.logger, penrs_http.httpx, etc.
#   since penrs_cache = utils and penrs_http = utils, these resolve to utils.*
penrs_cache = utils
penrs_http = utils

# Notebook-resident symbols loaded via IPython %run magic:
%run worker_nodes.ipynb    # brings PENRSWorker, truncate_for_context into scope
%run orchestrator.ipynb    # brings ArbiterAgent, ARBITER_SYSTEM_PROMPT,
                           #   run_all_workers, run_penrs into scope

# Router symbols
from utils import DOCUMENT_API_ROUTING, DocumentType, penrs_fetch_document
```

**Important:** If `httpx` is not installed in the kernel's Python environment, this cell fails at `import httpx` (a pre-existing environment gap — not a notebook regression).

#### Cell 4 — Arbiter Tests (4 functions)
Tests the `ArbiterAgent` class from `orchestrator.ipynb`:
1. `test_system_prompt_contains_required_role_and_mandatory_contradictions` — asserts "Lead Portfolio Manager", "Lipstick on a Pig", "Bailing Out", "Dilute and Delay" are all present in `ARBITER_SYSTEM_PROMPT`
2. `test_arbiter_validates_required_worker_schema_fields` — passes a worker missing `worker.weight`; asserts `pytest.raises(ValueError, match="worker.weight")`
3. `test_arbiter_clamps_scores_and_applies_star_rating_weights` — two workers with scores 1.8 and -2.4; asserts normalized to ±1.0, effective weights computed from star ratings, final weighted score correct
4. `test_arbiter_returns_json_schema_with_contradiction_flags_and_severities` — worker summary containing "lipstick on a pig" and "bailing out"; asserts `contradictions` list has correct `flagged` and `severity` values

#### Cell 6 — Cache Tests (6 functions)
All use `_reload_cache_module(monkeypatch, cache_dir)` which does `importlib.reload(utils)` to pick up a fresh `PENRS_CACHE_DIR` from the monkeypatched env var. Returns the reloaded `utils` module as `mod`.
1. `test_cache_key_is_deterministic` — same inputs → same 64-char hex; different date → different key
2. `test_cache_set_writes_json_with_metadata` — calls `cache_set`, reads the file, asserts `_cached_at`, `_api`, `_ticker`, `_doc_type`, `payload` keys
3. `test_cache_get_returns_payload_when_fresh` — set then get within `max_age_hours=12`
4. `test_cache_get_returns_none_for_missing_file` — get on non-existent entry; asserts returns None and logs at least once
5. `test_cache_get_returns_none_when_expired` — writes a cache file, backdates `_cached_at` by 5 hours, calls get with `max_age_hours=1`; asserts None
6. `test_cache_operations_are_logged` — asserts `logger.info` called ≥3 times during a set+get cycle

#### Cell 8 — Pipeline Tests (3 functions + 3 stub classes)
Stub classes: `StubWorker`, `StubArbiter`, `StubMaster` — injectable fakes with configurable return values.
1. `test_run_penrs_executes_pipeline_and_saves_report` — runs `run_penrs()` with 2 stub workers; asserts arbiter received 2 results, master received correct arbiter result, JSON report written to `penrs_reports/` directory
2. `test_run_penrs_worker_failure_is_isolated` — one worker raises `RuntimeError`, one succeeds; asserts failing worker has `status=="error"`, succeeding worker has `status=="available"`, report still saved
3. `test_run_all_workers_uses_asyncio_gather_return_exceptions` — uses `monkeypatch.setattr(asyncio, "gather", fake_gather)` to intercept the `asyncio.gather` call; asserts `return_exceptions=True` was passed

#### Cell 10 — Rate Limit Tests (6 functions)
All use `_reload_rate_limit_module()` which does `importlib.reload(utils)` + `utils._reset_rate_limit_state()`. Returns `utils` as `mod`.
1. `test_alpha_vantage_blocks_after_25_daily_calls` — makes 26 calls; asserts first 25 return True, 26th returns False
2. `test_alpha_vantage_sleeps_12_seconds_after_5_calls_same_minute` — patches `mod.time.sleep`; 5 calls in same minute succeed; 6th triggers sleep and returns True
3. `test_daily_counter_resets_at_midnight_boundary` — pre-populates state with 25/day count on previous day; patches `_now_utc` to midnight; asserts next call returns True and daily_count resets to 1
4. `test_minute_counter_resets_on_new_minute_for_sec_edgar` — 10 calls in one minute (all True), 11th returns False; advance time 1 minute; 12th returns True again
5. `test_other_apis_use_configurable_rpm_limit` — calls pubmed with `rpm_limit=2`; first two True, third False
6. `test_warning_logged_when_limits_approached_or_hit` — patches `mod.logger.warning`; asserts ≥2 warning calls during approach + hit

#### Cell 12 — Router Tests (5 functions)
1. `test_document_type_is_strict_enum_with_nine_values` — asserts `len(DocumentType) == 9`, routing table covers all 9 types, all API names are non-empty strings
2. `test_penrs_fetch_document_requires_document_type_enum` — passing `"sec_10q"` (string) raises `TypeError`
3. `test_penrs_fetch_document_aggregates_multi_source_results` — provides two fetchers for `BIOMEDICAL_EVIDENCE`; asserts both sources appear in `data.sources`
4. `test_penrs_fetch_document_returns_not_released_with_attempted_apis` — fetcher returns `{"status": "not_released"}`; asserts top-level `status == "not_released"` and `data.apis_attempted`
5. `test_penrs_fetch_document_handles_partial_failures` — one fetcher succeeds, one raises RuntimeError; asserts `status=="available"` with one source and one partial_failure

#### Cell 14 — Step 1 Tests / MCP Server Tests (4 functions with `@pytest.fixture`)
Uses `importlib.reload(penrs_mcp_server)` to re-initialize the server with fresh env vars.
1. `test_env_defaults` — with all env vars deleted, asserts ALPHA_VANTAGE_API_KEY=="demo", paths match expected defaults
2. `test_dirs_created` — sets custom PENRS_CACHE_DIR and PENRS_LOG_DIR; reloads; asserts both directories exist
3. `test_api_key_from_env` — sets ALPHA_VANTAGE_API_KEY="TESTKEY123"; reloads; asserts key matches
4. `test_mcp_server_named` — reloads; asserts `mcp.name == "penrs_mcp"`

#### Cell 16 — Step 2 Tests / HTTP Tests (8 functions)
All use `_patch_async_client(monkeypatch, planned_results)` which replaces `utils.httpx.AsyncClient` with a `FakeAsyncClient` that pops responses from a `planned_results` list.
1. `test_api_request_success_json_uses_default_timeout` — 200 + JSON; asserts result, timeout=30.0, correct call params
2. `test_api_request_returns_text_when_json_parse_fails` — 200 + text with json_error; asserts `{"text": "..."}`
3. `test_api_request_returns_structured_http_error_and_logs` — 500; asserts `{"error": "HTTP 500", "detail": "..."}` and `log_error` called
4. `test_api_request_retries_429_503_with_exponential_backoff` — 429, 503, 200; patches `utils.asyncio.sleep`; asserts sleeps=[1, 2], 3 client calls, 2 warning logs
5. `test_api_request_stops_after_max_retries_on_429` — 4×429; asserts 4 calls, sleeps=[1,2,4], result is HTTP 429 error
6. `test_api_request_timeout_is_user_friendly_and_logged` — raises `httpx.TimeoutException`; asserts `{"error": "Request timed out"}` and `log_error`
7. `test_api_request_request_error_is_user_friendly_and_logged` — raises `httpx.RequestError`; asserts `{"error": "Request failed", "detail": "network unreachable"}`
8. `test_api_request_respects_custom_timeout` — passes `timeout=5.5`; asserts `clients[0].timeout == 5.5`

#### Cell 18 — Worker Tests (5 functions)
1. `test_truncate_for_context_preserves_start_and_end` — 166-char string truncated to 60; asserts starts with 'A', ends with 'Z', contains `"[truncated"`, total length == 60
2. `test_parse_json_response_handles_markdown_block` — response with ` ```json {...} ``` `; asserts correct dict extracted
3. `test_parse_json_response_handles_embedded_prose_and_invalid_json_fallback` — JSON embedded in prose parsed correctly; non-JSON string returns `{"parse_error": "unable_to_parse_json"}`
4. `test_run_fetches_rubric_and_document_and_enriches_metadata` — full `run()` with injected fake rubric/document/LLM; asserts rubric fetched once, document fetched once, prompt contains "Document excerpt:", result has correct worker metadata and parsed LLM output
5. `test_run_handles_not_released_without_llm_call` — document fetcher returns `not_released`; asserts `status=="not_released"`, `apis_attempted` populated, LLM never called

#### Cell 20 — Test Runner
Calls 16 test functions directly (no fixture injection needed). Tests requiring `monkeypatch` are documented as requiring `ipytest`. Prints `PASS`/`FAIL` per test and a summary line.

---

## Data Flow (End to End)

```
run_penrs(ticker, date_from, date_to)            orchestrator.ipynb
    │
    ├─► run_all_workers([Worker1 ... Worker9])    orchestrator.ipynb
    │       │                                     asyncio.gather(return_exceptions=True)
    │       └─► PENRSWorker.run()                 worker_nodes.ipynb
    │               ├─► rubric_fetcher(id)        reads rubrics.json (not yet written)
    │               ├─► penrs_fetch_document()    utils.py
    │               │       └─► fetcher(api) × N  asyncio.gather
    │               │               └─► _api_request()         utils.py
    │               │                       ├─► _check_rate_limit()  utils.py
    │               │                       ├─► httpx GET
    │               │                       └─► cache_get/set()      utils.py
    │               └─► llm_invoker(prompt)       calls Claude (not yet wired)
    │
    ├─► _evaluate_with_arbiter(arbiter, results)  orchestrator.ipynb
    │       └─► ArbiterAgent.evaluate()
    │               ├─► validate schema
    │               ├─► normalize to [-1, 1]
    │               ├─► apply star-rating weights
    │               └─► detect contradiction patterns
    │
    ├─► MasterAgent.synthesize()                  orchestrator.ipynb
    │       └─► compute final_score, worker counts
    │
    └─► PENRSReport.model_validate(payload)       orchestrator.ipynb (pydantic)
            └─► write JSON → penrs_reports/{filename}.json
```

---

## Import Graph (Post Step 9)

```
penrs_mcp_server.py
  imports: os, logging, pathlib, dotenv, fastmcp
  (no project imports)

utils.py
  imports: asyncio, hashlib, json, logging, os,
           threading, time, datetime, enum, pathlib, typing
           httpx  ← only third-party dep

worker_nodes.ipynb
  imports: utils (DocumentType, penrs_fetch_document)
           json, re, pathlib, typing  ← stdlib only

orchestrator.ipynb
  imports: asyncio, json, re, datetime, pathlib, typing  ← stdlib
           pydantic (BaseModel, ConfigDict, Field)        ← third-party

tests.ipynb (cell-02)
  imports: utils
           httpx, pytest                                  ← third-party
           %run worker_nodes.ipynb → PENRSWorker, truncate_for_context
           %run orchestrator.ipynb → ArbiterAgent, ARBITER_SYSTEM_PROMPT,
                                      run_all_workers, run_penrs
```

**No circular dependencies.** Import graph is a strict DAG: `utils.py` has no project imports; notebooks depend on `utils.py` and each other only via `%run`.

---

## Spec Compliance Matrix (Post Step 9)

| Spec Step | Title | Status | Test Location |
|---|---|---|---|
| Step 1 (E1-F1-S1) | FastMCP Server Init | ✅ Complete | `tests.ipynb` Cell 14 (4 tests) |
| Step 2 (E1-F2-S1) | Async HTTP Client | ✅ Complete | `tests.ipynb` Cell 16 (8 tests) |
| Step 3 (E1-F3-S1) | File-Based Cache | ✅ Complete | `tests.ipynb` Cell 6 (6 tests) |
| Step 4 (E1-F4-S1) | Per-API Rate Limiter | ✅ Complete | `tests.ipynb` Cell 10 (6 tests) |
| Step 9 (E3-F7-S1) | Unified Document Router | ✅ Complete | `tests.ipynb` Cell 12 (5 tests) |
| Step 11 (E5-F1-S1) | PENRSWorker Base Class | ✅ Complete | `tests.ipynb` Cell 18 (5 tests) |
| Step 14 (E6-F1-S1) | Arbiter Agent | ✅ Complete | `tests.ipynb` Cell 4 (4 tests) |
| Step 16 (E8-F1-S1) | Pipeline Orchestration | ✅ Complete | `tests.ipynb` Cell 8 (3 tests) |

**Total: 41 test functions across 8 sections.**

---

## Environment Variables

| Variable | Default | Used In | Purpose |
|---|---|---|---|
| `ALPHA_VANTAGE_API_KEY` | `"demo"` | `penrs_mcp_server.py` | Alpha Vantage market data API key |
| `PENRS_CACHE_DIR` | `".penrs_cache"` | `utils.py` (Cache section) | Absolute path for JSON cache files |
| `PENRS_LOG_DIR` | `".penrs_logs"` | `penrs_mcp_server.py` | Directory for `penrs.log` file |
| `PENRS_DEFAULT_RPM_LIMIT` | `"60"` | `utils.py` (Rate Limit section) | RPM cap for APIs not explicitly configured |
| `SEC_USER_AGENT` | — | Future: SEC EDGAR fetcher | Required header for EDGAR API calls |
| `ANTHROPIC_API_KEY` | — | Future: LLM invoker | For Claude API calls from workers |
| `OPENFDA_API_KEY` | — | Future: openFDA fetcher | openFDA endpoint key |
| `NCBI_API_KEY` | — | Future: PubMed fetcher | PubMed E-utilities key |

---

## What Is Not Yet Implemented

The complete infrastructure is in place. The following are the remaining work items (from `implementation_plan.md`, deferred steps):

| Step | Description | Blocker |
|---|---|---|
| Step D1 | Pydantic rubric models, `rubrics.json`, `penrs_get_rubric`/`penrs_list_rubrics` MCP tools | Need rubric schema design |
| Step D2 | Alpha Vantage fetchers: Earnings, Form 4, Sentiment, Price History | Need real API integration |
| Step D3 | SEC EDGAR fetchers: Search, Fetch, CIK resolution | Need real API integration |
| Step D4 | ClinicalTrials.gov, openFDA, PubMed fetchers | Need real API integration |
| Step D5 | Workers 1–9: concrete `PENRSWorker` instances with real `rubric_id`s | Need D1–D4 + LLM wiring |
| Step D6 | Master programmatic scoring (PENRS formula) + LLM report synthesis | Need scoring formula |
| Step D7 | `calibration_events.json` + `compute_abnormal_return()` vs SPY | Need calibration data |
| Step D8 | Calibration run, PENRS vs outcomes comparison, sensitivity matrix | Need D7 |
| Step D9 | Display cells for Sections A/B/C/D in notebooks | Need D5–D6 |
| Step D10 | `universe.json` + Alpha Vantage coverage verification | Need D2 |
| Step D11 | Company IR Scraper (URL registry, pipeline fetch, PDF download) | New |
| Step D12 | Programmatic diffing (Clinical Trial, Pipeline page) | New |
| Step D13 | `penrs_cache_clear`, `penrs_cache_stats` MCP tools | Straightforward to add |
| Step D14 | Historical tracking (`{ticker}_history.json`), 0.3-point drift alerts | Need D5–D6 |

**What "real API fetchers" means:** Each fetcher is an async callable matching the `Fetcher` type alias — `async def fetch(ticker: str, date_range: tuple, doc_type: DocumentType) -> dict`. Register it in `penrs_fetch_document`'s `fetchers` argument (or in a wrapper around `penrs_fetch_document`). The routing table and retry/cache/rate-limit infrastructure already handles everything else.

---

## Known Environment Gaps (Pre-existing, Not Regressions)

These cause test failures in the current dev environment and are pre-existing issues from before Step 9:

1. **`httpx` not installed** in the system Python 3.9.6 path used by the `pytest` CLI → `test_step2.py` (now `tests.ipynb` Cell 16) collection errors / import cascade in `tests.ipynb` Cell 2
2. **`pydantic` not installed** → `test_pipeline.py` (now `tests.ipynb` Cell 8) collection errors; `orchestrator.ipynb` Cell 2 fails if `pydantic` missing
3. **`python-dotenv` not installed** → `test_step1.py` (now `tests.ipynb` Cell 14) runtime failures
4. **Python 3.9.6 `int | None` syntax** → `X | Y` union syntax requires Python 3.10+; `from __future__ import annotations` (used in notebooks) works around this at definition time but not always at test collection time

The Jupyter kernel uses Python 3.10+ so notebook cells defining functions execute correctly; the issues surface only when running pytest against the old `.py` files or when the import cell cascades. All 41 test functions are structurally correct for Python 3.10+.
