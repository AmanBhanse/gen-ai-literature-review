# papers/sources/semantic_scholar.py
"""Semantic Scholar paper source implementation."""

import time
import requests
from papers.sources.base import BasePaperSource, Paper
from papers.constants import (
    SEMANTIC_SCHOLAR_SEARCH_ENDPOINT,
    SEMANTIC_SCHOLAR_FIELDS,
    MIN_REQUEST_DELAY,
)


class SemanticScholarSource(BasePaperSource):
    """Fetch papers from Semantic Scholar REST API."""
    
    def fetch(self, query: str, num_papers: int = 1) -> list:
        """
        Fetch papers from Semantic Scholar using the official REST API.
        Single attempt (no retries) - falls back to next source on failure.
        
        Endpoint: GET https://api.semanticscholar.org/graph/v1/paper/search
        
        Args:
            query: Search query
            num_papers: Number of papers to fetch
            
        Returns:
            list[Paper]: List of papers or empty list on failure
        """
        papers = []
        try:
            simplified_query = self._simplify_query(query)
            self.logger.info(f"Searching Semantic Scholar REST API for: '{simplified_query}'")
            
            # Add delay to respect rate limits
            time.sleep(MIN_REQUEST_DELAY)
            
            params = {
                "query": simplified_query,
                "limit": num_papers,
                "fields": SEMANTIC_SCHOLAR_FIELDS
            }
            
            response = requests.get(SEMANTIC_SCHOLAR_SEARCH_ENDPOINT, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("data", [])
                
                if not results:
                    self.logger.debug("Semantic Scholar API returned no results")
                    return []
                
                for paper in results:
                    if paper.get("title"):
                        papers.append(Paper(
                            title=paper.get("title", "No title"),
                            summary=paper.get("abstract", "No abstract available"),
                            link=paper.get("url", "No link available"),
                            source="semantic_scholar",
                            year=paper.get("year"),
                            citations=paper.get("citationCount")
                        ))
                
                if papers:
                    self.logger.info(f"✓ Found {len(papers)} paper(s) via Semantic Scholar API")
                return papers
                
            else:
                self.logger.debug(f"Semantic Scholar API returned {response.status_code}, will try fallback")
                return []
                
        except requests.Timeout:
            self.logger.debug("Semantic Scholar API request timed out")
            return []
        except Exception as e:
            self.logger.debug(f"Semantic Scholar API error: {type(e).__name__}: {str(e)[:100]}")
            return []
