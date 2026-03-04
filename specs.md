# Product Specification: Step 1 (E1-F1-S1) â€” FastMCP Server Initialization
## Complete: true
## Goal
Initialize the FastMCP server named 'penrs_mcp' with environment variable loading for API keys, cache, and log directories.
## Requirements
- Server must start via `python penrs_mcp_server.py` using the stdio transport.
- Load `ALPHA_VANTAGE_API_KEY` from the environment or use a default fallback.
- Configure `PENRS_CACHE_DIR` and `PENRS_LOG_DIR` via environment variables.
## Success Criteria
- Server starts without error when run with `python penrs_mcp_server.py`.
- `ALPHA_VANTAGE_API_KEY` loaded from env or falls back to default.
- `PENRS_CACHE_DIR` and `PENRS_LOG_DIR` configurable via env vars.
- Cache and log directories auto-created on startup.

# Product Specification: Step 2 (E1-F2-S1) â€” Async HTTP Client & Error Handling
## Complete: true
## Goal
Implement `_api_request()` as a shared async HTTP client using `httpx` with configurable timeouts and automatic retry logic.
## Requirements
- Support configurable timeout (30s default).
- Implement automatic retry on 429/503 HTTP status codes with exponential backoff (max 3 retries).
- Map HTTP exceptions to user-friendly, actionable error messages.
## Success Criteria
- `_api_request()` accepts url, params, headers, and api_name.
- Returns structured dict on success (parsed JSON or `{text: ...}`).
- Returns `{error: ..., detail: ...}` on HTTP errors with status code.
- Retries up to 3 times on 429/503 with exponential backoff.
- Timeout at 30s returns `{error: 'Request timed out'}`.
- All errors logged via the server logger.

# Product Specification: Step 3 (E1-F3-S1) â€” File-Based Caching System
## Complete: true
## Goal
Implement `cache_get()` and `cache_set()` using SHA-256 hashed file keys to prevent redundant API calls.
## Requirements
- Generate deterministic hash keys from `(api, ticker, doc_type, date)`.
- Store payloads locally as JSON with a `_cached_at` timestamp.
- Implement TTL-based expiration (`max_age_hours`).
## Success Criteria
- `cache_key()` produces a deterministic hash.
- `cache_set()` writes JSON with `_cached_at`, `_api`, `_ticker`, `_doc_type`, and payload.
- `cache_get()` returns payload if file exists and age < `max_age_hours`.
- `cache_get()` returns None if file is missing or expired.
- All cache operations are logged.

# Product Specification: Step 4 (E1-F4-S1) â€” Per-API Rate Limiter
## Complete: true
## Goal
Implement `_check_rate_limit()` to track daily and per-minute request counts per API to avoid burning quota.
## Requirements
- Enforce Alpha Vantage limits: 25/day, 5/minute.
- Enforce SEC EDGAR limits: 10/minute.
- Support configurable RPM limits for all other APIs.
- Auto-reset counters on new day/minute boundaries.
## Success Criteria
- Alpha Vantage blocks after 25 daily calls and returns False.
- Alpha Vantage sleeps 12s after 5 calls in the same minute.
- Daily counter resets at midnight boundary.
- Minute counter resets on new minute.
- Warning logged when limits are approached or hit.

# Product Specification: Step 9 (E3-F7-S1) â€” Unified Document Router
## Complete: true
## Goal
Create `penrs_fetch_document`, a master routing tool that accepts document requests and dispatches them to the correct underlying API tools.
## Requirements
- Route based on `(ticker, document_type, date_range)`.
- **CRITICAL:** Enforce Python `Enum` or `Literal` types for `document_type` to guarantee strict chain of custody.
- Aggregate multi-source results into a single standardized response.
## Success Criteria
- Routes all 9 document types to their correct API(s).
- Aggregates results from multiple APIs into a single response.
- Returns standardized `{status: 'available'|'not_released', data: {...}}`.
- `not_released` payload includes a list of APIs attempted.
- Handles partial failures gracefully (some APIs succeed, others fail).

# Product Specification: Step 11 (E5-F1-S1) â€” PENRSWorker Base Class
## Complete: true
## Goal
Create a reusable LangChain base class for all 9 worker agents to standardize execution logic and context management.
## Requirements
- Implement `run(ticker, date_from, date_to)` method.
- Fetch rubric via MCP/JSON and fetch document via the Unified Document Router.
- Truncate large documents to fit the Claude context window while preserving the beginning and end.
- Robustly parse the LLM's JSON response (handling markdown blocks and embedded prose).
## Success Criteria
- Automatic rubric and document retrieval.
- Standardized `not_released` response handling if documents are missing.
- JSON response parsing never throws a hard exception.
- Result is enriched with worker metadata (name, weight, signal_density).

# Product Specification: Step 14 (E6-F1-S1) â€” Arbiter Agent
## Complete: true
## Goal
Implement the Arbiter agent to validate worker responses and detect cross-document narrative divergence.
## Requirements
- Programmatically normalize all worker scores to [-1.0, 1.0] and apply star-rating weights.
- System prompt (Lead Portfolio Manager) must flag mandatory contradictions: "Lipstick on a Pig", "Bailing Out", and "Dilute and Delay".
## Success Criteria
- Validates each worker result has required schema fields.
- Clamps scores accurately and computes weighted scores.
- Output schema is valid JSON returning contradiction flags with assigned severity levels (High/Medium).

# Product Specification: Step 16 (E8-F1-S1) â€” Pipeline Orchestration
## Complete: true
## Goal
Implement `run_penrs()`, the single end-to-end orchestration function connecting the full agentic flow.
## Requirements
- Execute workers concurrently using `asyncio.gather` with error isolation.
- Pass aggregated worker results to the Arbiter, then to the Master.
- Save the final validated report to disk.
## Success Criteria
- `run_penrs(ticker, date_from, date_to)` executes the full pipeline without hanging.
- A failure in Worker X does not crash Worker Y.
- Full report is successfully saved as a structured JSON file to the `penrs_reports/` directory.
- Function returns the complete parsed report dictionary.


