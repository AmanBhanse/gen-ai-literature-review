# config.py
"""
Backward compatibility module for configuration.

This module re-exports settings and constants from the new config_module/
structure to maintain backward compatibility with existing code.

New code should prefer direct imports from config_module:
- from config_module import Settings, Constants
"""

# Import from new config_module structure
from config_module import Settings, Constants, default_constants

# Re-export as module-level variables for backward compatibility
GROQ_API_KEY = Settings.GROQ_API_KEY
DEBUG_MODE = Settings.DEBUG_MODE
PAPER_SOURCE = Settings.PAPER_SOURCE
ENABLE_PAPER_CACHE = Settings.ENABLE_PAPER_CACHE
SEMANTIC_SCHOLAR_API_KEY = Settings.SEMANTIC_SCHOLAR_API_KEY
OPENALEX_API_KEY = Settings.OPENALEX_API_KEY

# Export word counts
LITERATURE_REVIEW_WORD_COUNT = default_constants.LITERATURE_REVIEW_WORD_COUNT
SINGLE_PAPER_SUMMARY_WORD_COUNT = default_constants.SINGLE_PAPER_SUMMARY_WORD_COUNT

__all__ = [
    "GROQ_API_KEY",
    "DEBUG_MODE",
    "PAPER_SOURCE",
    "ENABLE_PAPER_CACHE",
    "SEMANTIC_SCHOLAR_API_KEY",
    "OPENALEX_API_KEY",
    "LITERATURE_REVIEW_WORD_COUNT",
    "SINGLE_PAPER_SUMMARY_WORD_COUNT",
    "Settings",
    "Constants",
]
