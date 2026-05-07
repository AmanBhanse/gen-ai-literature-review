# papers/sources/base.py
"""Abstract base class for paper sources."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging


@dataclass
class Paper:
    """Represents a research paper."""
    title: str
    summary: str
    link: str
    source: str
    year: int = None
    citations: int = None
    authors: str = None


class BasePaperSource(ABC):
    """Abstract base for all paper sources."""
    
    def __init__(self, rate_limit: float = 5.0):
        self.rate_limit = rate_limit
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def fetch(self, query: str, num_papers: int = 1) -> list:
        """
        Fetch papers from source.
        
        Args:
            query: Search query
            num_papers: Number of papers to fetch
            
        Returns:
            list[Paper]: List of paper results
        """
        pass
    
    def _simplify_query(self, query: str) -> str:
        """
        Extract key terms from a query (removes author names and special chars).
        Long queries (like full paper titles) cause API timeouts.
        """
        import re
        
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
        self.logger.debug(f"Query simplified: '{query[:60]}...' → '{result}'")
        return result
