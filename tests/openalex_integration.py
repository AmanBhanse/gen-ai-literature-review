#!/usr/bin/env python3
"""
OpenAlex paper fetching implementation
Add this to utils.py to use OpenAlex as a paper source
"""

import requests
import os
import logging

logger = logging.getLogger(__name__)

OPENALEX_BASE_URL = "https://api.openalex.org"
OPENALEX_WORKS_ENDPOINT = f"{OPENALEX_BASE_URL}/works"
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "")

def fetch_papers_openalex(query: str, num_papers: int = 1) -> list:
    """
    Fetch papers from OpenAlex API
    
    Requires:
    - Free API key from https://openalex.org/settings/api
    - Set OPENALEX_API_KEY in .env file
    
    Args:
        query: Search query
        num_papers: Number of papers to fetch
    
    Returns:
        list: Papers with title, summary, link, source
    """
    if not OPENALEX_API_KEY:
        logger.debug("OpenAlex API key not configured (OPENALEX_API_KEY env var)")
        return []
    
    papers = []
    try:
        simplified_query = query[:100]  # Limit query length
        logger.info(f"Searching OpenAlex for: '{simplified_query}'")
        
        params = {
            "search": simplified_query,
            "per_page": num_papers,
            "api_key": OPENALEX_API_KEY
        }
        
        response = requests.get(OPENALEX_WORKS_ENDPOINT, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                logger.debug("OpenAlex returned no results")
                return []
            
            for paper in results:
                if paper.get("title"):
                    # Get first author if available
                    authors = []
                    for authorship in paper.get("authorships", [])[:3]:
                        author_name = authorship.get("author", {}).get("display_name", "Unknown")
                        authors.append(author_name)
                    
                    authors_str = ", ".join(authors) if authors else "Unknown"
                    
                    papers.append({
                        "title": paper.get("title", "No title"),
                        "summary": paper.get("abstract", "No abstract available"),
                        "link": paper.get("id", "No link available"),
                        "source": "openalex",
                        "year": paper.get("publication_year", "N/A"),
                        "citations": paper.get("cited_by_count", 0),
                        "authors": authors_str
                    })
            
            if papers:
                logger.info(f"✓ Found {len(papers)} paper(s) via OpenAlex")
            
            return papers[:num_papers]
            
        elif response.status_code == 401:
            logger.debug("OpenAlex API: Invalid API key (401)")
            return []
        elif response.status_code == 429:
            logger.debug("OpenAlex API: Rate limited (429)")
            return []
        else:
            logger.debug(f"OpenAlex API returned {response.status_code}")
            return []
            
    except requests.Timeout:
        logger.debug("OpenAlex API request timed out")
        return []
    except Exception as e:
        logger.debug(f"OpenAlex API error: {type(e).__name__}: {str(e)[:100]}")
        return []

# Example usage:
if __name__ == "__main__":
    # Test the function
    papers = fetch_papers_openalex("machine learning deep learning", num_papers=3)
    for i, paper in enumerate(papers, 1):
        print(f"\nPaper {i}:")
        print(f"  Title: {paper['title']}")
        print(f"  Authors: {paper['authors']}")
        print(f"  Year: {paper['year']}")
        print(f"  Citations: {paper['citations']}")
        print(f"  Link: {paper['link']}")
