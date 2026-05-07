# papers/fetcher.py
"""Main paper fetching router."""

import logging
from papers.sources import (
    ArxivSource,
    SemanticScholarSource,
    OpenAlexSource,
    GoogleScholarSource,
)
from papers.cache import CacheManager

logger = logging.getLogger(__name__)


class PaperFetcher:
    """Main API for fetching papers from multiple sources."""
    
    def __init__(self, config=None, cache_enabled: bool = True):
        """
        Initialize paper fetcher.
        
        Args:
            config: Configuration object with PAPER_SOURCE and API keys
            cache_enabled: Whether to use caching
        """
        self.config = config
        self.cache_enabled = cache_enabled
        self.cache = CacheManager() if cache_enabled else None
        
        # Initialize sources
        openalex_key = None
        if config:
            openalex_key = getattr(config, 'OPENALEX_API_KEY', None)
        
        self.sources = {
            "arxiv": ArxivSource(),
            "semantic_scholar": SemanticScholarSource(),
            "openalex": OpenAlexSource(api_key=openalex_key),
            "google_scholar": GoogleScholarSource(),
        }
    
    def fetch(self, query: str, num_papers: int = 1, source: str = None) -> list:
        """
        Fetch papers from configured source.
        
        Uses cache if available and enabled.
        
        Args:
            query: Search query
            num_papers: Number of papers to fetch
            source: Optional source override (uses config.PAPER_SOURCE if not provided)
            
        Returns:
            list[Paper]: List of papers or empty list on failure
        """
        # Check cache
        if self.cache_enabled and self.cache:
            if cached := self.cache.get(query):
                return cached
        
        # Determine source
        source_name = source
        if not source_name and self.config:
            source_name = getattr(self.config, 'PAPER_SOURCE', 'google_scholar')
        elif not source_name:
            source_name = 'google_scholar'
        
        if source_name not in self.sources:
            logger.error(f"Unknown source '{source_name}'. Valid options: {list(self.sources.keys())}")
            return []
        
        logger.info(f"Fetching papers from: {source_name}")
        
        # Convert Paper objects to dicts for caching
        papers_obj = self.sources[source_name].fetch(query, num_papers)
        papers = [self._paper_to_dict(p) for p in papers_obj]
        
        if papers:
            if self.cache_enabled and self.cache:
                self.cache.save(query, papers)
        else:
            logger.warning(f"✗ No papers found from {source_name} for: '{query}'")
        
        return papers
    
    @staticmethod
    def _paper_to_dict(paper) -> dict:
        """Convert Paper dataclass to dict for JSON serialization."""
        return {
            "title": paper.title,
            "summary": paper.summary,
            "link": paper.link,
            "source": paper.source,
            "year": paper.year,
            "citations": paper.citations,
            "authors": paper.authors,
        }


# Module-level convenience function
_fetcher = None


def fetch_papers(query: str, num_papers: int = 1, source: str = None) -> list:
    """
    Fetch papers from configured source (convenience function).
    
    Args:
        query: Search query
        num_papers: Number of papers to fetch
        source: Optional source override
        
    Returns:
        list: List of paper dicts or empty list on failure
    """
    global _fetcher
    if _fetcher is None:
        # Import here to avoid circular dependency
        from config import PAPER_SOURCE, ENABLE_PAPER_CACHE, OPENALEX_API_KEY
        
        class SimpleConfig:
            pass
        
        cfg = SimpleConfig()
        cfg.PAPER_SOURCE = PAPER_SOURCE
        cfg.OPENALEX_API_KEY = OPENALEX_API_KEY
        
        _fetcher = PaperFetcher(config=cfg, cache_enabled=ENABLE_PAPER_CACHE)
    
    return _fetcher.fetch(query, num_papers, source)
