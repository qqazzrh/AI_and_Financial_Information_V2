# PENRS Product Specifications & Audit Trail Architecture

This document dictates the sequential, test-driven build sequence for the PENRS financial intelligence pipeline. The CLI factory loop reads this top-to-bottom. Do not skip steps. Do not assume imports.

---

## Spec 1: Ingestion Engine & Immutable Cache (MCP Server)
**Step Reference:** D2, D3, D4
## Complete: true

**Goal:** Build the raw data ingestion layer, securely route external API calls, and lock the raw responses to disk to create an immutable ground truth for downstream auditing.

**Requirements:**
**1.1 File Structure & Imports:**
* **Target:** `penrs_mcp_server.py`.
* Import `_api_request`, `cache_set`, and `PENRS_CACHE_DIR` from the `utils` module.
* Ensure `os`, `logging`, and `FastMCP` are present.

**1.2 Alpha Vantage Fetcher:**
* Create `async def fetch_alpha_vantage(ticker: str, function: str, date: str | None = None) -> dict:`.
* Decorate with `@mcp.tool()`.
* **Logic:** Construct URL `https://www.alphavantage.co/query`. 
* **Params:** `{"function": function, "symbol": ticker, "apikey": ALPHA_VANTAGE_API_KEY}`.
* **Execution:** `result = await _api_request(url, params=params, api_name="alpha_vantage")`.
* **Cache Binding:** If `"error" not in result`, call `cache_set(api="alpha_vantage", ticker=ticker, doc_type=function, date=date, payload=result)`. Return `result`.

**1.3 SEC EDGAR Fetcher:**
* Create `async def fetch_sec_edgar(ticker: str, accession_number: str, primary_document: str) -> dict:`.
* Decorate with `@mcp.tool()`.
* **Logic:** URL `https://www.sec.gov/Archives/edgar/data/{ticker}/{accession_number}/{primary_document}`. (Using ticker as a proxy for CIK in MVP).
* **Headers:** `{"User-Agent": SEC_USER_AGENT}`.
* **Execution:** `result = await _api_request(url, headers=headers, api_name="sec_edgar")`.
* **Cache Binding:** If `"error" not in result`, call `cache_set(api="sec_edgar", ticker=ticker, doc_type="filing", date=None, payload=result)`.

**1.4 openFDA Fetcher:**
* Create `async def fetch_openfda(ticker: str, limit: int = 10) -> dict:`.
* Decorate with `@mcp.tool()`.
* **Logic:** URL `https://api.fda.gov/drug/event.json`.
* **Params:** `{"search": f"patient.drug.medicinalproduct:{ticker}", "limit": limit}`. Conditionally add `api_key` if `OPENFDA_API_KEY` exists.
* **Execution:** `await _api_request` with `api_name="openfda"`.
* **Cache Binding:** If `"error" not in result`, `cache_set` with `doc_type="adverse_events"`.

**1.5 PubMed Fetcher:**
* Create `async def fetch_pubmed(term: str, retmax: int = 5) -> dict:`.
* Decorate with `@mcp.tool()`.
* **Logic:** URL `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi`.
* **Params:** `{"db": "pubmed", "term": term, "retmode": "json", "retmax": retmax}`. Conditionally add `api_key` if `NCBI_API_KEY` exists.
* **Execution:** `await _api_request` with `api_name="pubmed"`.
* **Cache Binding:** If `"error" not in result`, `cache_set` using `ticker=term.replace(" ", "_")` and `doc_type="publications"`.

---

## Spec 2: Strict JSON Schema Injection for Worker Nodes
**Step Reference:** D1, D5
## Complete: true

**Goal:** Force the LLM to output highly structured, auditable evidence strings.

**Requirements:**
**2.1 Prompt Engineering:**
* **Target:** `PENRSWorker.build_prompt()` in `worker_nodes.ipynb`.
* **Action:** Append the following explicit instruction to the returned string:
    `"You MUST return a strictly valid JSON object adhering to this exact schema:\n"`
    `"{\n"`
    `"  \"score\": <float between -1.0 and 1.0>,\n"`
    `"  \"thesis\": <string>,\n"`
    `"  \"evidence_nodes\": [\n"`
    `"    {\"verbatim_quote\": <exact substring from document>, \"reasoning\": <string>}\n"`
    `"  ]\n"`
    `"}\n"`
    `"The 'verbatim_quote' MUST be an exact, character-for-character substring of the provided Document excerpt."`

**2.2 Parser Hardening:**
* **Target:** `PENRSWorker.parse_json_response()`.
* **Action:** After successfully extracting the JSON dict (via `json.loads` or `_extract_json_from_text`), programmatically enforce the schema.
* Ensure `"score"` exists and is a float. Ensure `"thesis"` is a string. Ensure `"evidence_nodes"` is a list. If any are missing, inject default empty values (`{"score": 0.0, "thesis": "Parse failure", "evidence_nodes": []}`) to prevent Orchestrator crashes.

**2.3 LLM Invocation:**
* **Target:** `PENRSWorker.__init__`.
* **Action:** Ensure the default or injected `llm_invoker` is prepared to interface with Anthropic's Messages API, specifically passing `system="Respond only in valid JSON."`.

---

## Spec 3: Verbatim Quote Validation & Arbiter Integration
**Step Reference:** D6
## Complete: true

**Goal:** Programmatically destroy LLM hallucinations and map validated quotes to the final report.

**Requirements:**
**3.1 Hallucination Destruction:**
* **Target:** `PENRSWorker.run()`.
* **Action:** After `parsed = self.parse_json_response(llm_raw)`, iterate over `parsed.get("evidence_nodes", [])`.
* **Validation:** For each node, verify `node["verbatim_quote"] in excerpt`. (Where `excerpt` is the output of `truncate_for_context`).
* **Pruning:** If `in` evaluates to `False`, drop the node entirely.
* **Score Override:** If `len(parsed["evidence_nodes"]) == 0` after pruning, forcibly set `parsed["score"] = 0.0` and append to `thesis`: `"[SYSTEM NOTE: Score neutralized due to hallucinated evidence.]"`.

**3.2 Arbiter Validation Override:**
* **Target:** `ArbiterAgent._validate_worker_result()` in `orchestrator.ipynb`.
* **Action:** Add a check: if `result.get("score", 0.0) != 0.0`, assert that `result.get("evidence_nodes")` exists and has `len() > 0`. Raise `ValueError` if a worker tries to pass a non-neutral score without validated evidence.

**3.3 Data Model Expansion:**
* **Target:** `PENRSReport` Pydantic class in `orchestrator.ipynb`.
* **Action:** Add `evidence: list[dict[str, Any]] = Field(default_factory=list)` to the class.

**3.4 Master Synthesis Mapping:**
* **Target:** `MasterAgent.synthesize()` in `orchestrator.ipynb`.
* **Action:** Iterate through `worker_results` where `status == "available"`. Extract all `evidence_nodes`. 
* **Cache Mapping:** For each node, inject the `cache_key`. Use `utils.cache_key(api="[derive_from_worker]", ticker=ticker, doc_type="[derive_from_worker]")`.
* Return the aggregated `evidence` list inside the master dictionary so it saves to the JSON report.

---

## Spec 4: Retro TUI Implementation (The Auditor)
**Step Reference:** New UI Layer
## Complete: false



**Goal:** Build the Tron-aesthetic, human-in-the-loop review interface using `Textual`.

**Requirements:**
**4.1 Framework Initialization:**
* **Target:** Create `penrs_tui.py` and `penrs_tui.tcss`.
* `class PenrsAuditor(App):` loading `CSS_PATH = "penrs_tui.tcss"`. Define `BINDINGS = [("q", "quit", "Quit")]`.

**4.2 CSS Design System (`penrs_tui.tcss`):**
* `Screen { background: #050510; }`
* `DirectoryTree, VerticalScroll, RichLog { border: double #00FFFF; padding: 1; }`
* `DirectoryTree:focus, VerticalScroll:focus, RichLog:focus { border: double #39FF14; }`
* `.flagged { border: double #FF003C; animation: pulse 2s linear infinite; }`
* `@keyframes pulse { 0% { border: double #FF003C; } 50% { border: double #4A0000; } 100% { border: double #FF003C; } }`
* `RichLog { color: #39FF14; }`

**4.3 Layout Composition:**
* **Target:** `compose(self)` method in `penrs_tui.py`.
* Yield a `Header()`.
* Yield a `Horizontal` container holding:
    * `DirectoryTree("./penrs_reports", id="ledger", classes="pane")`
    * `Vertical` container holding:
        * `VerticalScroll(id="synthesis", classes="pane")` (containing a `Label(id="master_score")`, `Label(id="contradictions")`, and `ListView(id="evidence_list")`)
        * `RichLog(id="ground_truth", classes="pane", wrap=True)`

**4.4 Reactive Event Handlers:**
* **Target:** `@on(DirectoryTree.FileSelected)`
* **Action:** Read the selected `.json` report from `penrs_reports/`. Parse it. Update `#master_score` with `master.final_score`. If `arbiter.contradictions` has flagged items, `add_class("flagged")` to `#synthesis`. Clear `#evidence_list`. Populate `#evidence_list` with `ListItem` widgets containing the `verbatim_quote`, attaching the `cache_key` to the `ListItem.id` or a custom attribute.
* **Target:** `@on(ListView.Selected)`
* **Action:** Extract the `cache_key`. Locate the file in `PENRS_CACHE_DIR` (`f"{key}.json"`). Extract the raw text. 
* **The Highlight:** Import `rich.text.Text`. Create a `Text` object of the raw document. Find the substring index of the `verbatim_quote`. Apply `style="bold black on #00FFFF"` exclusively to that substring. Write to `#ground_truth` `RichLog`.

---

## Spec 5: End-to-End Integration Test (The Audit Gate)
**Step Reference:** Final Validation
## Complete: false

**Goal:** Prove the pipeline works end-to-end, validating contradiction flags and quote extraction before allowing the loop to exit.

**Requirements:**
**5.1 Test Initialization:**
* **Target:** `tests.ipynb`.
* Create `def test_e2e_audit_trail_validation(monkeypatch, local_tmp_dir):`.

**5.2 Pipeline Mocking:**
* Create a mock `document_fetcher` returning:
    `{"status": "available", "data": "Q3 earnings were solid, however, we are severely delaying our phase 3 trials due to enrollment issues."}`
* Create a mock `rubric_fetcher` returning `{"criteria": "Test"}`.
* Create a mock `llm_invoker` returning a strict JSON string:
    `{"score": -0.9, "thesis": "Management is bailing out.", "evidence_nodes": [{"verbatim_quote": "we are severely delaying our phase 3 trials", "reasoning": "Classic delay."}]}`

**5.3 Execution:**
* Instantiate a `PENRSWorker` with these mocks.
* Run `asyncio.run(run_penrs(ticker="TEST", date_from="2026-01-01", date_to="2026-02-01", workers=[worker], report_dir=local_tmp_dir))`.

**5.4 Hard Assertions:**
* `assert report["master"]["final_score"] < 0.0`
* Extract `contradictions = {c["name"]: c for c in report["arbiter"]["contradictions"]}`.
* `assert contradictions["Dilute and Delay"]["flagged"] is True` (Note: The mock LLM said "bailing out" but the text says "delaying" - the Arbiter runs regex on the *summary*, so ensure the mock thesis or summary triggers the right flag). For this test, adjust mock `thesis` to trigger both if desired, e.g., `"Management is bailing out while pushing dilute and delay tactics."`
* `assert len(report["master"]["evidence"]) == 1`
* `assert report["master"]["evidence"][0]["verbatim_quote"] == "we are severely delaying our phase 3 trials"`
* Verify the report file was physically written to `local_tmp_dir` and contains these exact JSON structures.