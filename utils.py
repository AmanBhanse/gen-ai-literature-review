# utils.py
"""
Backward compatibility module for utility functions.

This module re-exports functions from the new utils/ submodule structure
to maintain backward compatibility with existing code.

New code should prefer direct imports from utils submodules:
- from utils.extraction import extract_draft_from_message
- from utils.query import simplify_query
- from papers import fetch_papers
"""

import logging
from config import DEBUG_MODE

# Configure logging based on DEBUG_MODE
log_level = logging.INFO if DEBUG_MODE else logging.WARNING
logging.basicConfig(level=log_level, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Import extraction functions for backward compatibility
from utils.extraction import (
    extract_draft_from_message,
    extract_draft_from_message_json,
    extract_draft_from_message_separator,
)

# Import paper fetching for backward compatibility
from papers import fetch_papers

# Expose simplify_query (backward compat alias)
from utils.query import simplify_query as _simplify_query

# Re-export for immediate access
__all__ = [
    "extract_draft_from_message",
    "extract_draft_from_message_json", 
    "extract_draft_from_message_separator",
    "fetch_papers",
]
