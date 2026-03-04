import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastmcp import FastMCP

load_dotenv()

# --- Configuration ---
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
PENRS_CACHE_DIR = Path(os.getenv("PENRS_CACHE_DIR", ".penrs_cache")).resolve()
PENRS_LOG_DIR = Path(os.getenv("PENRS_LOG_DIR", ".penrs_logs")).resolve()

# --- Directory setup ---
PENRS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    mcp.run(transport="stdio")
