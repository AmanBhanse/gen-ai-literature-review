# papers/cache.py
"""Cache management for paper fetches."""

import json
import re
import logging
from pathlib import Path
from papers.constants import CACHE_DIR_NAME

logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching of paper fetch results."""
    
    def __init__(self, cache_dir: str = CACHE_DIR_NAME):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, query: str) -> Path:
        """Generate cache file path for a query"""
        safe_query = re.sub(r'[^a-z0-9]', '_', query.lower())[:50]
        return self.cache_dir / f"{safe_query}.json"
    
    def get(self, query: str) -> list:
        """
        Load papers from cache if available.
        
        Args:
            query: Search query
            
        Returns:
            list: Cached papers or None if not found
        """
        cache_path = self._get_cache_path(query)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    papers = json.load(f)
                    logger.info(f"✓ Loaded {len(papers)} papers from cache for: '{query}'")
                    return papers
            except Exception as e:
                logger.debug(f"Cache read failed: {e}")
        return None
    
    def save(self, query: str, papers: list) -> None:
        """
        Save papers to cache.
        
        Args:
            query: Search query
            papers: List of papers to cache
        """
        try:
            cache_path = self._get_cache_path(query)
            with open(cache_path, 'w') as f:
                json.dump(papers, f, indent=2)
            logger.info(f"✓ Cached {len(papers)} papers for query: '{query}'")
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")
