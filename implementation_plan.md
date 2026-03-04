# Implementation Plan

## Sprint 1: Foundation (Core MCP & Rubrics)
*Goal: Server boots, handles cache/rate limits, and serves rubrics. Unit tests pass.*
- [x] **Step 1:** Initialize FastMCP server (`penrs_mcp_server.py`), `.env.example`, and `requirements.txt`.
- [x] **Step 2:** Implement `_api_request()` with httpx, exponential backoff, and standardized error formatting.
- [x] **Step 3:** Implement SHA-256 file-based caching and write `test_cache.py`.
- [x] **Step 4:** Implement stateful rate limiting (daily/minute tracking) and write `test_rate_limit.py`.
- [ ] **Step 5:** Define Pydantic models for rubrics, create `rubrics.json`, and implement `penrs_get_rubric` / `penrs_list_rubrics` tools. Write `test_rubrics.py`.

## Sprint 2: Data Layer (API Integrations & Routing)
*Goal: All APIs fetch successfully, and the router directs traffic correctly.*
- [ ] **Step 6:** Implement Alpha Vantage endpoints (Earnings, Form 4, Sentiment, Price) with standardized response validation.
- [ ] **Step 7:** Implement SEC EDGAR endpoints (Search, Fetch, CIK resolution).
- [ ] **Step 8:** Implement ClinicalTrials.gov, openFDA, and PubMed endpoints.
- [x] **Step 9:** Implement Unified Document Router (`penrs_fetch_document`). **CRITICAL:** Enforce Python `Enum` for `document_type` to guarantee strict chain of custody.
- [ ] **Step 10:** Write live integration smoke tests (`test_integration.py` and `test_router.py`) using ticker MRNA.

## Sprint 3: Agent Core (Workers & Synthesis)
*Goal: Agents can process documents and generate the final report schema.*
- [x] **Step 11:** Create `PENRSWorker` base class (handles routing, execution, and context truncation via `truncate_for_context`). Write unit tests for JSON parsing fallbacks.
- [ ] **Step 12:** Implement Workers 1 through 9.
- [ ] **Step 13:** Implement async `run_all_workers()` using `asyncio.gather(return_exceptions=True)`.
- [x] **Step 14:** Implement Arbiter programmatic pre-validation (score normalization to [-1, 1]) and LLM narrative divergence detection.
- [ ] **Step 15:** Implement Master programmatic scoring (PENRS formula in Python) and LLM report synthesis.
- [x] **Step 16:** Build `run_penrs()` end-to-end orchestration function and define the `PENRSReport` Pydantic schema for validation.

## Sprint 4: Validation & Display
*Goal: System is calibrated against ground truth and outputs readable reports.*
- [ ] **Step 17:** Create `calibration_events.json` (MRNA, BIIB, SRPT, SAVA) and implement `compute_abnormal_return()` against SPY.
- [ ] **Step 18:** Run calibration script, compare PENRS scores to actual stock outcomes, and generate sensitivity matrix.
- [ ] **Step 19:** Build Jupyter Notebook display cells to render Sections A, B, C, and D cleanly.
- [ ] **Step 20:** Create `universe.json` and build the Alpha Vantage coverage verification script.

## Sprint 5: Polish & Advanced Features
*Goal: IR scraping, diffing utilities, and historical tracking.*
- [ ] **Step 21:** Implement Company IR Scraper (URL registry, pipeline fetch, deck PDF download).
- [ ] **Step 22:** Implement programmatic diffing tools (Clinical Trial diffing, Pipeline page diffing).
- [ ] **Step 23:** Implement `penrs_cache_clear` and `penrs_cache_stats` MCP tools.
- [ ] **Step 24:** Implement historical tracking (`{ticker}_history.json`) and 0.3-point drift alert logic.

