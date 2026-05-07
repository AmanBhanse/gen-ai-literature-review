# utils.py
import re
import json
import logging
import os
import time
import requests
from pathlib import Path
from scholarly import scholarly
from config import DEBUG_MODE

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False

# Configure logging based on DEBUG_MODE
log_level = logging.INFO if DEBUG_MODE else logging.WARNING
logging.basicConfig(level=log_level, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Semantic Scholar API configuration
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_PAPER_SEARCH_ENDPOINT = f"{S2_API_BASE}/paper/search"
S2_FIELDS = "abstract,authors,citationCount,publicationDate,title,url,venue,year"

# Rate limiting constants - Semantic Scholar has strict limits (~100 requests/5 minutes for free tier)
MIN_REQUEST_DELAY = 5  # 5 seconds minimum between requests (more conservative)
MAX_RETRIES = 2  # Retry up to 2 times on rate limit
RETRY_DELAY = 10  # Start with 10 second delay on first 429

# Cache directory setup
CACHE_DIR = Path("paper_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _simplify_query(query: str) -> str:
    """
    Extract key terms from a query (removes author names and special chars).
    Long queries (like full paper titles) causes API timeouts.
    
    Examples:
    - "Transformers: Attention is All You Need by Vaswani" → "Transformers Attention"
    - "Advancements in Generative AI: ... Staphord Bengesi" → "Generative AI"
    """
    # Remove author names (typically after "by" or at the end after capital letter pattern)
    simplified = re.sub(r'\s+by\s+.+', '', query, flags=re.IGNORECASE)
    
    # Extract key words (capitalize words are often titles/concepts)
    words = simplified.split()
    
    # Keep significant words, limit length for API queries
    key_words = []
    for word in words:
        # Skip small words, special chars
        if len(word) > 3 and not re.search(r'[^a-zA-Z0-9\s\-]', word):
            key_words.append(word)
        if len(key_words) >= 5:  # Limit to 5 key terms
            break
    
    result = " ".join(key_words) if key_words else query[:50]
    logger.debug(f"Query simplified: '{query[:60]}...' → '{result}'")
    return result


def _get_cache_path(query: str) -> Path:
    """Generate cache file path for a query"""
    safe_query = re.sub(r'[^a-z0-9]', '_', query.lower())[:50]
    return CACHE_DIR / f"{safe_query}.json"


def _load_from_cache(query: str) -> list:
    """Load papers from cache if available"""
    cache_path = _get_cache_path(query)
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                papers = json.load(f)
                logger.info(f"✓ Loaded {len(papers)} papers from cache for: '{query}'")
                return papers
        except Exception as e:
            logger.debug(f"Cache read failed: {e}")
    return None


def _save_to_cache(query: str, papers: list) -> None:
    """Save papers to cache"""
    try:
        cache_path = _get_cache_path(query)
        with open(cache_path, 'w') as f:
            json.dump(papers, f, indent=2)
        logger.info(f"✓ Cached {len(papers)} papers for query: '{query}'")
    except Exception as e:
        logger.debug(f"Cache write failed: {e}")


def fetch_papers_semantic_scholar(query: str, num_papers: int = 1) -> list:
    """
    Fetch papers from Semantic Scholar using the official REST API.
    Single attempt (no retries) - falls back to next source on failure.
    
    Endpoint: GET https://api.semanticscholar.org/graph/v1/paper/search
    """
    papers = []
    try:
        simplified_query = _simplify_query(query)
        logger.info(f"Searching Semantic Scholar REST API for: '{simplified_query}'")
        
        # Add delay to respect rate limits
        time.sleep(MIN_REQUEST_DELAY)
        
        params = {
            "query": simplified_query,
            "limit": num_papers,
            "fields": S2_FIELDS
        }
        
        response = requests.get(S2_PAPER_SEARCH_ENDPOINT, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("data", [])
            
            if not results:
                logger.debug("Semantic Scholar API returned no results")
                return []
            
            for paper in results:
                if paper.get("title"):
                    papers.append({
                        "title": paper.get("title", "No title"),
                        "summary": paper.get("abstract", "No abstract available"),
                        "link": paper.get("url", "No link available"),
                        "source": "semantic_scholar"
                    })
            
            if papers:
                logger.info(f"✓ Found {len(papers)} paper(s) via Semantic Scholar API")
            return papers
            
        else:
            logger.debug(f"Semantic Scholar API returned {response.status_code}, will try fallback")
            return []
            
    except requests.Timeout:
        logger.debug("Semantic Scholar API request timed out")
        return []
    except Exception as e:
        logger.debug(f"Semantic Scholar API error: {type(e).__name__}: {str(e)[:100]}")
        return []


def fetch_papers_arxiv(query: str, num_papers: int = 1) -> list:
    """
    Fetch papers from arXiv (reliable for AI/ML papers, no rate limits)
    Simplifies queries to avoid rate limiting with long search terms.
    """
    if not ARXIV_AVAILABLE:
        logger.debug("arxiv not installed")
        return []
    
    papers = []
    try:
        # Simplify query to avoid rate limiting on long queries
        simplified_query = _simplify_query(query)
        logger.info(f"Searching arXiv for: '{simplified_query}'")
        
        # Add delay to respect rate limits
        time.sleep(MIN_REQUEST_DELAY)
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=simplified_query,
            max_results=num_papers,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        for paper in client.results(search):
            if len(papers) >= num_papers:
                break
            
            papers.append({
                "title": paper.title,
                "summary": paper.summary,
                "link": paper.entry_id,
                "source": "arxiv"
            })
        
        if papers:
            logger.info(f"✓ Found {len(papers)} paper(s) via arXiv")
    except Exception as e:
        logger.debug(f"arXiv search failed: {str(e)[:100]}")
    
    return papers


def fetch_papers_openalex(query: str, num_papers: int = 1) -> list:
    """
    Fetch papers from OpenAlex (comprehensive, no strict rate limits).
    Requires OPENALEX_API_KEY in .env file.
    """
    from config import OPENALEX_API_KEY
    
    if not OPENALEX_API_KEY:
        logger.error("OPENALEX_API_KEY not set in .env. Get a free key at https://openalex.org/settings/api")
        return []
    
    papers = []
    try:
        simplified_query = _simplify_query(query)
        logger.info(f"Searching OpenAlex for: '{simplified_query}'")
        
        # Add delay to respect rate limits
        time.sleep(MIN_REQUEST_DELAY)
        
        url = "https://api.openalex.org/works"
        params = {
            "search": simplified_query,
            "per_page": num_papers,
            "mailto": "user@example.com"  # OpenAlex requires this
        }
        headers = {"Authorization": f"Bearer {OPENALEX_API_KEY}"}
        
        response = requests.get(url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 401:
            logger.error("OpenAlex authentication failed - invalid API key")
            return []
        elif response.status_code == 429:
            logger.warning("OpenAlex rate limited")
            return []
        elif response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                logger.debug("OpenAlex returned no results")
                return []
            
            for work in results:
                if work.get("title"):
                    authors = ", ".join([a["author"]["display_name"] for a in work.get("authorships", [])[:3]])
                    papers.append({
                        "title": work.get("title", "No title"),
                        "summary": work.get("abstract") or work.get("title", "No abstract available"),
                        "link": work.get("id", "No link"),
                        "source": "openalex",
                        "year": work.get("publication_year"),
                        "citations": work.get("cited_by_count", 0),
                        "authors": authors
                    })
            
            if papers:
                logger.info(f"✓ Found {len(papers)} paper(s) via OpenAlex")
            return papers
        else:
            logger.debug(f"OpenAlex API returned {response.status_code}")
            return []
            
    except requests.Timeout:
        logger.debug("OpenAlex API request timed out")
        return []
    except Exception as e:
        logger.debug(f"OpenAlex API error: {type(e).__name__}: {str(e)[:100]}")
        return []


def fetch_papers_google_scholar(query: str, num_papers: int = 1) -> list:
    """
    Fetch papers from Google Scholar (fallback, less reliable)
    Includes rate limiting to avoid overloading the API.
    """
    papers = []
    try:
        # Add delay to respect rate limits
        time.sleep(MIN_REQUEST_DELAY)
        
        logger.info(f"Searching Google Scholar for: '{query}'")
        search_results = scholarly.search_pubs(query)
        for i, paper in enumerate(search_results):
            if i >= num_papers:
                break
            papers.append({
                "title": paper["bib"]["title"],
                "summary": paper["bib"].get("abstract", "No summary available"),
                "link": paper.get("pub_url", "No link available"),
                "source": "google_scholar"
            })
        
        if papers:
            logger.info(f"✓ Found {len(papers)} papers via Google Scholar")
    except Exception as e:
        logger.debug(f"Google Scholar search failed: {e}")
    
    return papers


def fetch_papers(query: str, num_papers: int = 1) -> list:
    """
    Fetch papers using the configured paper source (no fallback chain).
    Uses PAPER_SOURCE from config: "google_scholar", "arxiv", "semantic_scholar", or "openalex"
    
    Strategy:
    1. Check cache first (if enabled)
    2. Use selected paper source
    3. Save to cache if successful
    
    Returns:
        list: A list of dictionaries containing paper details (title, summary, link, source)
    """
    from config import PAPER_SOURCE
    
    # Check cache first
    cached_papers = _load_from_cache(query)
    if cached_papers:
        return cached_papers
    
    # Dispatch to selected paper source
    source_map = {
        "google_scholar": fetch_papers_google_scholar,
        "arxiv": fetch_papers_arxiv,
        "semantic_scholar": fetch_papers_semantic_scholar,
        "openalex": fetch_papers_openalex,
    }
    
    if PAPER_SOURCE not in source_map:
        logger.error(f"Unknown PAPER_SOURCE '{PAPER_SOURCE}'. Valid options: {list(source_map.keys())}")
        return []
    
    fetch_func = source_map[PAPER_SOURCE]
    logger.info(f"Fetching papers from: {PAPER_SOURCE}")
    
    papers = fetch_func(query, num_papers)
    if papers:
        _save_to_cache(query, papers)
    else:
        logger.warning(f"✗ No papers found from {PAPER_SOURCE} for: '{query}'")
    
    return papers


def extract_draft_from_message_json(last_message):
    """
    Extract content from JSON-formatted LLM response.
    Expected format: {"content": "actual text", "status": "complete"}
    
    Returns:
        tuple: (content, success, method)
    """
    if not last_message:
        return None, False, "no_message"
    
    message_content = last_message.content.strip()
    
    try:
        # Find JSON object in the message (handles case where LLM adds extra text)
        json_match = re.search(r'\{.*\}', message_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            
            if "content" in parsed and parsed["content"]:
                logger.info(f"✓ Successfully extracted via JSON method. Status: {parsed.get('status', 'unknown')}")
                return parsed["content"].strip(), True, "json"
    except (json.JSONDecodeError, AttributeError) as e:
        logger.debug(f"JSON extraction failed: {e}")
    
    return None, False, "json_failed"


def extract_draft_from_message_separator(last_message):
    """
    Extract content from separator-delimited LLM response.
    Fallback method for backward compatibility.
    
    Returns:
        tuple: (content, success, method)
    """
    if not last_message:
        return None, False, "no_message"
    
    try:
        message_content = last_message.content
        pattern = rf"OUTPUT : STARTS(.*?)OUTPUT : ENDS"
        match = re.search(pattern, message_content, re.DOTALL)
        
        if match:
            content = match.group(1).strip()
            if content:
                logger.info("✓ Successfully extracted via SEPARATOR method (fallback)")
                return content, True, "separator"
    except Exception as e:
        logger.debug(f"Separator extraction failed: {e}")
    
    return None, False, "separator_failed"


def extract_draft_from_message(last_message):
    """
    Extract content from LLM response using hybrid approach.
    Tries JSON first (preferred), falls back to separator format.
    
    Returns:
        str: Extracted content or None if all methods fail
    """
    if not last_message:
        logger.warning("No message received from LLM")
        return None
    
    # Try JSON method first (preferred)
    content, success, method = extract_draft_from_message_json(last_message)
    if success:
        return content
    
    # Fall back to separator method
    content, success, method = extract_draft_from_message_separator(last_message)
    if success:
        return content
    
    # Both methods failed - log warning and attempt to get any substantial content
    logger.warning(f"✗ Failed to extract content using standard methods. Message preview: {last_message.content[:100]}...")
    
    # Last resort: check if message is very short and return it directly
    if last_message.content and len(last_message.content) > 10:
        logger.warning("Returning raw message content as last resort")
        return last_message.content
    
    return None
