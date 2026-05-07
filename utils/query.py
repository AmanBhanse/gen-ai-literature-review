# utils/query.py
"""Query processing utilities."""

import re
import logging

logger = logging.getLogger(__name__)


def simplify_query(query: str) -> str:
    """
    Extract key terms from a query (removes author names and special chars).
    Long queries (like full paper titles) cause API timeouts.
    
    Examples:
    - "Transformers: Attention is All You Need by Vaswani" → "Transformers Attention"
    - "Advancements in Generative AI: ... Staphord Bengesi" → "Generative AI"
    
    Args:
        query: Raw search query
        
    Returns:
        str: Simplified query with key terms
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
