# config_module/constants.py
"""Static constants (never change at runtime)."""

from dataclasses import dataclass


@dataclass
class Constants:
    """Static constants for the application."""
    
    # Content generation word counts
    LITERATURE_REVIEW_WORD_COUNT: int = 500
    SINGLE_PAPER_SUMMARY_WORD_COUNT: int = 100
    
    # API rate limiting
    MIN_REQUEST_DELAY: float = 5.0  # seconds
    MAX_RETRIES: int = 2
    RETRY_DELAY: float = 10.0  # seconds
    
    # API Endpoints
    SEMANTIC_SCHOLAR_API_BASE: str = "https://api.semanticscholar.org/graph/v1"
    OPENALEX_API_BASE: str = "https://api.openalex.org/works"
    
    # Cache
    CACHE_DIR_NAME: str = "paper_cache"
    
    def __getitem__(self, key: str):
        """Allow dict-like access for backward compatibility."""
        return getattr(self, key)


# Global instance
default_constants = Constants()
