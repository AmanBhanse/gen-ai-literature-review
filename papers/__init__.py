# papers/__init__.py
"""Papers module - Fetch papers from multiple sources."""

from papers.fetcher import fetch_papers, PaperFetcher
from papers.sources import Paper

__all__ = [
    "fetch_papers",
    "PaperFetcher",
    "Paper",
]
