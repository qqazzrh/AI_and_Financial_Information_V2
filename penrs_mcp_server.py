import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastmcp import FastMCP
from utils import PENRS_CACHE_DIR, _api_request, cache_set

load_dotenv()

# --- Configuration ---
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "PENRS/1.0")
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
PENRS_LOG_DIR = Path(os.getenv("PENRS_LOG_DIR", ".penrs_logs")).resolve()

# --- Directory setup ---
PENRS_LOG_DIR.mkdir(parents=True, exist_ok=True)

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(PENRS_LOG_DIR / "penrs.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("penrs_mcp")
logger.info("PENRS MCP server starting up.")
logger.info("Cache dir: %s | Log dir: %s", PENRS_CACHE_DIR, PENRS_LOG_DIR)

# --- FastMCP server ---
mcp = FastMCP("penrs_mcp")

@mcp.tool()
async def fetch_alpha_vantage(
    ticker: str,
    function: str,
    date: str | None = None,
) -> dict:
    url = "https://www.alphavantage.co/query"
    params = {
        "function": function,
        "symbol": ticker,
        "apikey": ALPHA_VANTAGE_API_KEY,
    }
    result = await _api_request(url, params=params, api_name="alpha_vantage")
    if "error" not in result:
        cache_set(
            api="alpha_vantage",
            ticker=ticker,
            doc_type=function,
            date=date,
            payload=result,
        )
    return result


@mcp.tool()
async def fetch_sec_edgar(
    ticker: str,
    accession_number: str,
    primary_document: str,
) -> dict:
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{ticker}/{accession_number}/{primary_document}"
    )
    headers = {"User-Agent": SEC_USER_AGENT}
    result = await _api_request(url, headers=headers, api_name="sec_edgar")
    if "error" not in result:
        cache_set(
            api="sec_edgar",
            ticker=ticker,
            doc_type="filing",
            date=None,
            payload=result,
        )
    return result


@mcp.tool()
async def fetch_openfda(ticker: str, limit: int = 10) -> dict:
    url = "https://api.fda.gov/drug/event.json"
    params = {
        "search": f"patient.drug.medicinalproduct:{ticker}",
        "limit": limit,
    }
    if OPENFDA_API_KEY:
        params["api_key"] = OPENFDA_API_KEY

    result = await _api_request(url, params=params, api_name="openfda")
    if "error" not in result:
        cache_set(
            api="openfda",
            ticker=ticker,
            doc_type="adverse_events",
            date=None,
            payload=result,
        )
    return result


@mcp.tool()
async def fetch_pubmed(term: str, retmax: int = 5) -> dict:
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": retmax,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    result = await _api_request(url, params=params, api_name="pubmed")
    if "error" not in result:
        cache_set(
            api="pubmed",
            ticker=term.replace(" ", "_"),
            doc_type="publications",
            date=None,
            payload=result,
        )
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
