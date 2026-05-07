# papers/sources/google_scholar.py
"""Google Scholar paper source implementation."""

import time
from scholarly import scholarly
from papers.sources.base import BasePaperSource, Paper
from papers.constants import MIN_REQUEST_DELAY


class GoogleScholarSource(BasePaperSource):
    """Fetch papers from Google Scholar (fallback, less reliable)."""
    
    def fetch(self, query: str, num_papers: int = 1) -> list:
        """
        Fetch papers from Google Scholar.
        Includes rate limiting to avoid overloading the API.
        
        Args:
            query: Search query
            num_papers: Number of papers to fetch
            
        Returns:
            list[Paper]: List of papers or empty list on failure
        """
        papers = []
        try:
            # Add delay to respect rate limits
            time.sleep(MIN_REQUEST_DELAY)
            
            self.logger.info(f"Searching Google Scholar for: '{query}'")
            search_results = scholarly.search_pubs(query)
            for i, paper in enumerate(search_results):
                if i >= num_papers:
                    break
                papers.append(Paper(
                    title=paper["bib"]["title"],
                    summary=paper["bib"].get("abstract", "No summary available"),
                    link=paper.get("pub_url", "No link available"),
                    source="google_scholar"
                ))
            
            if papers:
                self.logger.info(f"✓ Found {len(papers)} papers via Google Scholar")
        except Exception as e:
            self.logger.debug(f"Google Scholar search failed: {e}")
        
        return papers
