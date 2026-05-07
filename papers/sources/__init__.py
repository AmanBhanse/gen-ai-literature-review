# papers/sources/__init__.py
"""Paper source implementations."""

from papers.sources.base import BasePaperSource, Paper
from papers.sources.arxiv import ArxivSource
from papers.sources.semantic_scholar import SemanticScholarSource
from papers.sources.openalex import OpenAlexSource
from papers.sources.google_scholar import GoogleScholarSource

__all__ = [
    "BasePaperSource",
    "Paper",
    "ArxivSource",
    "SemanticScholarSource",
    "OpenAlexSource",
    "GoogleScholarSource",
]
