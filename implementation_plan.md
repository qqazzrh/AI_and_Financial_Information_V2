# PENRS Implementation Plan & State Tracker

This document tracks the execution state of the PENRS Financial Intelligence Pipeline. The CLI factory loop updates this file automatically after the test suite passes.

**Status Legend:**
- [ ] Pending
- [x] Complete

---

## Phase 1: Ingestion Engine & Immutable Cache (Spec 1)
- [x] **Step 1.1:** Import `_api_request` and `cache_set` into `penrs_mcp_server.py`.
- [x] **Step 1.2:** Implement `@mcp.tool()` Alpha Vantage fetcher with cache binding.
- [x] **Step 1.3:** Implement `@mcp.tool()` SEC EDGAR fetcher with cache binding.
- [x] **Step 1.4:** Implement `@mcp.tool()` openFDA fetcher with cache binding.
- [x] **Step 1.5:** Implement `@mcp.tool()` PubMed fetcher with cache binding.

## Phase 2: Strict JSON Schema Injection (Spec 2)
- [x] **Step 2.1:** Update `PENRSWorker.build_prompt()` to demand the strict JSON schema (`score`, `thesis`, `evidence_nodes`, `verbatim_quote`).
- [x] **Step 2.2:** Update `PENRSWorker.parse_json_response()` to enforce the presence of required schema keys.
- [x] **Step 2.3:** Wire the Anthropic API call into the worker's `llm_invoker` forcing JSON mode.

## Phase 3: Verbatim Quote Validation (Spec 3)
- [x] **Step 3.1:** Implement string validation in `PENRSWorker.run()` to drop hallucinated quotes not found in the excerpt.
- [x] **Step 3.2:** Force worker score to 0.0 if all evidence nodes are dropped.
- [x] **Step 3.3:** Update `ArbiterAgent._validate_worker_result()` to reject non-neutral scores lacking valid evidence.
- [x] **Step 3.4:** Expand `PENRSReport` Pydantic model to include the `evidence` array.
- [x] **Step 3.5:** Update `MasterAgent.synthesize()` to extract valid evidence and append the source document `cache_key`.

## Phase 4: Retro TUI Implementation (Spec 4)
- [ ] **Step 4.1:** Create `penrs_tui.py` and `penrs_tui.tcss` with the Textual App shell and neon Tron color palette.
- [ ] **Step 4.2:** Build the Left Pane (`DirectoryTree`) to watch `.penrs_reports/`.
- [ ] **Step 4.3:** Build the Top Right Pane to display the synthesis, contradiction flags, and clickable evidence nodes.
- [ ] **Step 4.4:** Build the Bottom Right Pane to load `.penrs_cache/` files and invert-highlight the verbatim quotes.

## Phase 5: End-to-End Integration Test (Spec 5)
- [ ] **Step 5.1:** Create `test_e2e_audit_trail_validation` in `tests.ipynb`.
- [ ] **Step 5.2:** Mock the document fetcher with the planted "Dilute and Delay" string.
- [ ] **Step 5.3:** Execute `run_penrs()` with the mock constraints.
- [ ] **Step 5.4:** Assert the contradiction is flagged, the quote is extracted perfectly, and the report writes to disk cleanly.