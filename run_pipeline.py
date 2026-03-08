#!/usr/bin/env python3
"""Execute the full biotech disclosure pipeline for a given ticker.

This script loads all notebook code cells to initialize the pipeline,
then calls run_full_pipeline() for the specified ticker.
"""
import json
import sys
import os

# Ensure .env is loaded
from dotenv import load_dotenv
load_dotenv()

NB_PATH = os.path.join(os.path.dirname(__file__), "notebooks", "01_biotech_disclosure_pipeline.ipynb")

def extract_code_cells(nb_path: str) -> list[str]:
    """Extract source code from all code cells in the notebook."""
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)
    sources = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            sources.append(source)
    return sources


def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else "BIIB"
    print(f"=== Loading pipeline from notebook ({NB_PATH}) ===")

    code_cells = extract_code_cells(NB_PATH)

    # Build a shared namespace and execute all cells to load definitions
    ns = {"__name__": "__main__", "__builtins__": __builtins__}

    for i, code in enumerate(code_cells):
        try:
            exec(compile(code, f"<cell_{i}>", "exec"), ns)
        except Exception as e:
            print(f"  [cell {i}] Warning: {type(e).__name__}: {e}")

    print(f"\n=== Pipeline loaded. Running run_full_pipeline('{ticker}') ===\n")

    run_full_pipeline = ns["run_full_pipeline"]
    result = run_full_pipeline(ticker)

    # Print structured summary
    print("\n" + "=" * 70)
    print(f"PIPELINE RESULTS FOR {result['ticker']} ({result['company_name']})")
    print(f"Run ID: {result['run_id']}")
    print("=" * 70)

    # Retrieval summary
    print("\n-- RETRIEVAL --")
    for doc_type, rr in result["retrieval_results"].items():
        candidate_title = rr.selected_candidate.title[:80] if rr.selected_candidate else "None"
        print(f"  {doc_type.value:30s} | {rr.status.value:20s} | {candidate_title}")

    # Worker summary
    print("\n-- WORKER ANALYSIS --")
    for doc_type, wo in result["worker_outputs"].items():
        sentiment = wo.sentiment.label.value if wo.sentiment else "N/A"
        confidence = f"{wo.confidence:.2f}" if wo.confidence is not None else "N/A"
        print(f"  {doc_type.value:30s} | {wo.status.value:15s} | sentiment={sentiment:10s} | confidence={confidence}")

    # Arbiter summary
    ao = result["arbiter_output"]
    print("\n-- ARBITER --")
    print(f"  Status:        {ao.status.value}")
    print(f"  Sentiment:     {ao.sentiment.label.value if ao.sentiment else 'N/A'}")
    print(f"  Conflicts:     {len(ao.conflicting_signals)}")
    print(f"  Uncertainties: {len(ao.unresolved_uncertainties)}")

    # Final payload summary
    fp = result["final_payload"]
    print("\n-- FINAL UI PAYLOAD --")
    print(f"  Status:           {fp.status.value}")
    print(f"  Overall Sentiment: {fp.overall_sentiment_label.value if fp.overall_sentiment_label else 'N/A'}")
    print(f"  Disclosures:      {len(fp.disclosures)}")
    print(f"  Missing Types:    {[dt.value for dt in fp.missing_document_types]}")

    # Write full JSON payload to file
    output_path = os.path.join(os.path.dirname(__file__), f"output_{ticker.upper()}_payload.json")
    payload_json = fp.model_dump(mode="json")
    with open(output_path, "w") as f:
        json.dump(payload_json, f, indent=2, default=str)
    print(f"\n  Full FinalUIPayload JSON written to: {output_path}")

    # Also write full result summary
    summary_path = os.path.join(os.path.dirname(__file__), f"output_{ticker.upper()}_summary.json")
    summary = {
        "ticker": result["ticker"],
        "company_name": result["company_name"],
        "run_id": result["run_id"],
        "retrieval": {
            doc_type.value: {
                "status": rr.status.value,
                "selected_candidate_id": rr.selection_decision.selected_candidate_id,
                "selected_title": rr.selected_candidate.title if rr.selected_candidate else None,
                "source_url": rr.selected_candidate.source_url if rr.selected_candidate else None,
                "issues": [str(issue.error_code) for issue in rr.issues],
            }
            for doc_type, rr in result["retrieval_results"].items()
        },
        "workers": {
            doc_type.value: {
                "status": wo.status.value,
                "sentiment_label": wo.sentiment.label.value if wo.sentiment else None,
                "confidence": wo.confidence,
                "warning_count": len(wo.warnings),
                "issue_count": len(wo.issues),
            }
            for doc_type, wo in result["worker_outputs"].items()
        },
        "arbiter": {
            "status": ao.status.value,
            "sentiment_label": ao.sentiment.label.value if ao.sentiment else None,
            "conflict_count": len(ao.conflicting_signals),
            "uncertainty_count": len(ao.unresolved_uncertainties),
        },
        "final_payload": payload_json,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Full result summary written to: {summary_path}")


if __name__ == "__main__":
    main()
