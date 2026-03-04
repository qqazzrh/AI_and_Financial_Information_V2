# PENRS System Readout
**Generated:** 2026-03-02
**Test Results:** 41/41 passed (1.86s)
**Spec Coverage:** 6/6 implemented specs validated

---

## What Is PENRS?

PENRS is a multi-agent financial intelligence pipeline designed to analyze publicly traded biotech/pharma companies. It fetches and scores nine types of financial and scientific documents, runs them through specialized worker agents, and produces a weighted composite score with narrative contradiction detection. The name implies a scoring/rating system ("PENRS score") intended to signal whether a stock is positioned as bullish or bearish based on cross-document evidence.

The architecture is: **Data Layer → Worker Agents → Arbiter → Master → Report on disk**.

---

## Component Reference

### 1. `penrs_mcp_server.py` — Server Entrypoint
**Spec:** Step 1 (E1-F1-S1)

Initializes the FastMCP server named `penrs_mcp`. This is the outer shell that other tools register onto (the MCP tool registry). On startup it:

- Loads environment variables from `.env` via `python-dotenv`
- Reads `ALPHA_VANTAGE_API_KEY` (falls back to `"demo"`)
- Reads `PENRS_CACHE_DIR` (falls back to `.penrs_cache/`) and `PENRS_LOG_DIR` (falls back to `.penrs_logs/`)
- Auto-creates both directories (`mkdir(parents=True, exist_ok=True)`)
- Configures dual-channel logging: file handler to `penrs.log` + stdout stream
- Instantiates `FastMCP("penrs_mcp")`
- Runs via `mcp.run(transport="stdio")` when executed directly

**Usage:** `python penrs_mcp_server.py`

---

### 2. `penrs_http.py` — Shared Async HTTP Client
**Spec:** Step 2 (E1-F2-S1)

A single `async def _api_request(url, params, headers, api_name, timeout)` function. All API calls in the system route through this function. Key behaviors:

- Uses `httpx.AsyncClient` with a 30s default timeout
- Retries automatically on **HTTP 429** (rate limited) and **503** (server unavailable) with exponential backoff: 1s → 2s → 4s (max 3 retries)
- On timeout: returns `{"error": "Request timed out", "detail": "URL: ..."}`
- On non-retryable HTTP error: returns `{"error": "HTTP <code>", "detail": "<first 500 chars of body>"}`
- On success: returns parsed JSON dict, or `{"text": <raw text>}` if body is not JSON
- On retry exhaustion: returns `{"error": "Max retries exceeded", ...}`
- All errors logged via the `penrs_mcp` logger

---

### 3. `penrs_cache.py` — SHA-256 File-Based Cache
**Spec:** Step 3 (E1-F3-S1)

A deterministic key-value cache that stores API responses as JSON files, preventing redundant API calls. Three public functions:

**`cache_key(api, ticker, doc_type, date)`**
Hashes the string `"{api}|{ticker}|{doc_type}|{date}"` with SHA-256 and returns the hex digest. Deterministic — same inputs always produce the same key.

**`cache_set(api, ticker, doc_type, date, payload)`**
Writes a JSON file to `PENRS_CACHE_DIR/{sha256}.json` with structure:
```json
{
  "_cached_at": "<ISO 8601 UTC timestamp>",
  "_api": "...",
  "_ticker": "...",
  "_doc_type": "...",
  "_date": "...",
  "payload": { ... }
}
```

**`cache_get(api, ticker, doc_type, date, max_age_hours)`**
Reads the cache file and checks freshness. Returns `payload` dict if the file exists and `age < max_age_hours`. Returns `None` on miss, expired entry, or corrupt file. Includes a backward-compatible fallback for legacy files written without a `"payload"` wrapper key.

---

### 4. `penrs_rate_limit.py` — Per-API Rate Limiter
**Spec:** Step 4 (E1-F4-S1)

A thread-safe stateful rate limiter (`threading.Lock`) tracking per-API daily and per-minute request counts in an in-process dict `_RATE_LIMIT_STATE`. One public function:

**`_check_rate_limit(api_name, rpm_limit=None) -> bool`**

Hardcoded limits by API:
| API | Per-Minute | Per-Day | Behavior at Limit |
|-----|-----------|---------|-------------------|
| Alpha Vantage (`alpha`, `alpha_vantage`, `alphavantage`) | 5 | 25 | Sleeps 12s on minute limit; returns `False` on daily limit |
| SEC EDGAR (`sec`, `sec_edgar`, `edgar`) | 10 | None | Returns `False` |
| All others | `PENRS_DEFAULT_RPM_LIMIT` env var (default 60) | None | Returns `False` |

Counters auto-reset when the day or minute boundary changes. Emits `WARNING` logs when a limit is being approached (1 call away) or hit.

---

### 5. `penrs_router.py` — Unified Document Router
**Spec:** Step 9 (E3-F7-S1)

The routing hub. Defines the full document taxonomy as a Python `Enum` and dispatches fetch requests to the correct API(s) concurrently.

**`DocumentType` (Enum)**
Exactly 9 values:
```
EARNINGS_CALL, FORM_4, NEWS_SENTIMENT, PRICE_HISTORY,
SEC_10K, SEC_10Q, SEC_8K, CLINICAL_TRIALS, BIOMEDICAL_EVIDENCE
```

**`DOCUMENT_API_ROUTING` (routing table)**
Maps each `DocumentType` to one or more API names:
- `EARNINGS_CALL`, `FORM_4`, `NEWS_SENTIMENT`, `PRICE_HISTORY` → `alpha_vantage`
- `SEC_10K`, `SEC_10Q`, `SEC_8K` → `sec_edgar`
- `CLINICAL_TRIALS` → `clinicaltrials_gov`
- `BIOMEDICAL_EVIDENCE` → `openfda` + `pubmed` (multi-source)

**`penrs_fetch_document(ticker, document_type, date_range, fetchers)`**
The master routing function. Accepts a `fetchers` dict mapping API name → async callable. Dispatches all required APIs concurrently via `asyncio.gather(return_exceptions=True)`. Returns a standardized envelope:
- `{"status": "available", "data": {..., "sources": [...]}}` — if at least one API succeeded
- `{"status": "not_released", "data": {..., "errors": [...]}}` — if all APIs failed

Partial failures are preserved in `data.partial_failures` even when some sources succeed. Enforces strict `isinstance(document_type, DocumentType)` type guard — passing a raw string raises `TypeError`.

---

### 6. `penrs_worker.py` — PENRSWorker Base Class
**Spec:** Step 11 (E5-F1-S1)

A reusable base class that standardizes the full execution lifecycle for all 9 worker agents. Each worker instance is configured with its identity and injected dependencies (rubric fetcher, document fetcher, LLM invoker).

**Constructor parameters:**
- `name` — worker identifier (e.g., `"EarningsCallWorker"`)
- `weight` — relative importance in final scoring (float, 0–1)
- `signal_density` — confidence in signal quality (float, 0–1); drives star rating
- `rubric_id` — key to look up in `rubrics.json`
- `document_type` — `DocumentType` enum value this worker processes
- `rubric_fetcher` — defaults to reading `rubrics.json` from disk
- `document_fetcher` — defaults to `penrs_fetch_document`
- `llm_invoker` — defaults to a stub that returns `"{}"`
- `max_context_chars` — truncation limit for document excerpts (default: 12,000 chars)

**`truncate_for_context(text, max_chars)`**
Truncates large documents while preserving head and tail: splits budget evenly between the first and last portions, inserting a `...[truncated N chars]...` marker in the middle.

**`parse_json_response(response)`**
Never raises. Tries four strategies in order:
1. Direct dict passthrough
2. `json.loads(text)`
3. Extract from fenced code block (` ```json ... ``` `)
4. Regex scan for the first `{` or `[` and attempt partial decode

Falls back to `{"parse_error": "unable_to_parse_json", "raw_response": text}`.

**`run(ticker, date_from, date_to)`**
Main execution method:
1. Fetches rubric via `rubric_fetcher`
2. Fetches document via `document_fetcher`
3. If document is `not_released`, returns early with status metadata
4. Coerces document to string, truncates, builds structured prompt
5. Invokes LLM, parses JSON response
6. Returns enriched result dict with `worker` metadata block

---

### 7. `penrs_arbiter.py` — Arbiter Agent
**Spec:** Step 14 (E6-F1-S1)

The programmatic scoring layer. Validates all worker outputs, normalizes scores, applies star-rating weights, and runs narrative contradiction detection. No LLM is called — this is pure Python logic.

**`ArbiterAgent.evaluate(worker_results)`**

For each worker result with `status == "available"`:
1. Validates presence of `status`, `worker.name`, `worker.weight`, `worker.signal_density`, `result.score`
2. Normalizes raw score to `[-1.0, 1.0]` via `max(-1.0, min(1.0, score))`
3. Derives star rating (1–5) from `signal_density`:
   - ≥0.85 → 5★, ≥0.65 → 4★, ≥0.45 → 3★, ≥0.25 → 2★, else 1★
4. Computes `effective_weight = base_weight × (star_rating / 5.0)`
5. Accumulates `weighted_score_sum` and `total_effective_weight`
6. Final `weighted_score = weighted_score_sum / total_effective_weight`, re-clamped to `[-1.0, 1.0]`

**Contradiction detection** scans aggregated `summary`, `thesis`, `narrative`, `analysis` text fields for three mandatory patterns (case-insensitive regex):
| Name | Severity | Triggers On |
|------|----------|-------------|
| Lipstick on a Pig | High | `lipstick on a pig` |
| Bailing Out | High | `bailing out`, `bail-out`, `bailout` |
| Dilute and Delay | Medium | `dilute and delay` |

Returns a list of 3 contradiction objects, each with `name`, `severity`, `flagged: bool`, `evidence: str | null`.

---

### 8. `penrs_pipeline.py` — End-to-End Orchestration
**Spec:** Step 16 (E8-F1-S1)

The top-level orchestration layer. Wires workers → arbiter → master → disk.

**`run_penrs(ticker, date_from, date_to, workers, arbiter, master, report_dir, now)`**

1. **Worker execution:** Calls `run_all_workers()` which uses `asyncio.gather(return_exceptions=True)` — a crash in Worker N does not affect Worker M
2. **Arbiter:** Filters to only `status == "available"` results, passes to `ArbiterAgent.evaluate()`. If no valid results, returns `weighted_score: 0.0` and empty lists
3. **Master:** `MasterAgent.synthesize()` computes a simple programmatic summary: `final_score` (the arbiter's `weighted_score`), `available_worker_count`, `total_worker_count`
4. **Serialization:** Validates the complete report payload via the `PENRSReport` Pydantic model, then serializes and saves to `penrs_reports/{ticker}_{date_from}_{date_to}_{timestamp}Z.json`
5. Returns the full parsed report dict

**`PENRSReport` (Pydantic model)**
Schema for the saved report file:
```
ticker, date_from, date_to, generated_at,
worker_results[], arbiter{}, master{}, report_path
```
`model_config = ConfigDict(extra="allow")` — extra fields do not cause validation failures.

---

## Data Flow (End to End)

```
run_penrs(ticker, date_from, date_to)
    │
    ├─► run_all_workers([Worker1 ... Worker9])       asyncio.gather
    │       │
    │       └─► PENRSWorker.run(ticker, date_from, date_to)
    │               ├─► rubric_fetcher(rubric_id)    reads rubrics.json
    │               ├─► penrs_fetch_document(...)    penrs_router.py
    │               │       └─► fetcher(api) × N     asyncio.gather
    │               │               └─► _api_request(url, ...)   penrs_http.py
    │               │                       └─► [cache_get / cache_set]  penrs_cache.py
    │               │                       └─► [_check_rate_limit]      penrs_rate_limit.py
    │               └─► llm_invoker(prompt)
    │
    ├─► ArbiterAgent.evaluate(worker_results)
    │       ├─► validate schema
    │       ├─► normalize scores to [-1, 1]
    │       ├─► apply star-rating weights
    │       └─► detect contradiction patterns
    │
    ├─► MasterAgent.synthesize(...)
    │       └─► compute final_score, worker counts
    │
    └─► PENRSReport.model_validate(payload)
            └─► write JSON to penrs_reports/{filename}.json
```

---

## Spec Compliance Matrix

| Step | Title | Status | Test File | Tests |
|------|-------|--------|-----------|-------|
| Step 1 | FastMCP Server Init | ✅ Complete | `test_step1.py` | 4 passed |
| Step 2 | Async HTTP Client | ✅ Complete | `test_step2.py` | 8 passed |
| Step 3 | File-Based Cache | ✅ Complete | `test_cache.py` | 6 passed |
| Step 4 | Per-API Rate Limiter | ✅ Complete | `test_rate_limit.py` | 6 passed |
| Step 9 | Unified Document Router | ✅ Complete | `test_router.py` | 5 passed |
| Step 11 | PENRSWorker Base Class | ✅ Complete | `test_worker.py` | 5 passed |
| Step 14 | Arbiter Agent | ✅ Complete | `test_arbiter.py` | 4 passed |
| Step 16 | Pipeline Orchestration | ✅ Complete | `test_pipeline.py` | 3 passed |

**Total: 41/41 tests pass.**

---

## What Is Not Yet Implemented

Per `implementation_plan.md`, the following are still pending (no source code or tests exist for them):

| Step | Description |
|------|-------------|
| Step 5 | Pydantic rubric models, `rubrics.json`, `penrs_get_rubric` / `penrs_list_rubrics` MCP tools |
| Step 6 | Alpha Vantage API endpoints (Earnings, Form 4, Sentiment, Price) |
| Step 7 | SEC EDGAR endpoints (Search, Fetch, CIK lookup) |
| Step 8 | ClinicalTrials.gov, openFDA, PubMed endpoints |
| Step 10 | Live integration smoke tests against real ticker (MRNA) |
| Step 12 | Workers 1–9 concrete implementations |
| Step 13 | `run_all_workers()` integration (already in pipeline; concrete workers missing) |
| Step 15 | Master programmatic scoring (PENRS formula) + LLM report synthesis |
| Steps 17–24 | Calibration, Jupyter display, coverage verification, IR scraper, diffing tools, cache management, historical tracking |

The current codebase is the **complete foundation layer**. The infrastructure handles everything correctly — rate limiting, caching, HTTP retries, routing, worker lifecycle, arbitration, and report serialization. What remains is wiring real API fetchers and real LLM calls into the workers.

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ALPHA_VANTAGE_API_KEY` | `demo` | Alpha Vantage market data |
| `PENRS_CACHE_DIR` | `.penrs_cache/` | Cache file storage |
| `PENRS_LOG_DIR` | `.penrs_logs/` | Log file storage |
| `PENRS_DEFAULT_RPM_LIMIT` | `60` | Default RPM for unconfigured APIs |
| `SEC_USER_AGENT` | — | Required for SEC EDGAR requests |
| `ANTHROPIC_API_KEY` | — | For Claude LLM invocations (workers) |
| `OPENFDA_API_KEY` | — | openFDA endpoint |
| `NCBI_API_KEY` | — | PubMed E-utilities |
