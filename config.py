# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable is not set. "
        "Please create a .env file in the project root with: GROQ_API_KEY=your_api_key"
    )

# Logging Configuration
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Paper Search Configuration
# Choose ONE paper source: "google_scholar", "arxiv", "semantic_scholar", or "openalex"
# Note: No fallback chain - uses the selected source only
PAPER_SOURCE = os.getenv("PAPER_SOURCE", "google_scholar")

# Enable paper caching (local JSON files to avoid repeated API calls)
ENABLE_PAPER_CACHE = os.getenv("ENABLE_PAPER_CACHE", "true").lower() == "true"

# Optional API Keys
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", None)  # For Semantic Scholar (not required)
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", None)  # For OpenAlex (required if PAPER_SOURCE=openalex)

# Word Counts
LITERATURE_REVIEW_WORD_COUNT = 500
SINGLE_PAPER_SUMMARY_WORD_COUNT = 100
