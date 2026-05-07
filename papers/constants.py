# papers/constants.py
"""Constants for paper fetching (API endpoints, rate limits)."""

# Semantic Scholar API configuration
SEMANTIC_SCHOLAR_API_BASE = "https://api.semanticscholar.org/graph/v1"
SEMANTIC_SCHOLAR_SEARCH_ENDPOINT = f"{SEMANTIC_SCHOLAR_API_BASE}/paper/search"
SEMANTIC_SCHOLAR_FIELDS = "abstract,authors,citationCount,publicationDate,title,url,venue,year"

# Rate limiting constants (shared across all sources)
# Semantic Scholar has strict limits (~100 requests/5 minutes for free tier)
MIN_REQUEST_DELAY = 5  # 5 seconds minimum between requests
MAX_RETRIES = 2  # Retry up to 2 times on rate limit
RETRY_DELAY = 10  # Start with 10 second delay on first 429

# OpenAlex API configuration
OPENALEX_API_BASE = "https://api.openalex.org/works"

# Cache directory
CACHE_DIR_NAME = "paper_cache"
