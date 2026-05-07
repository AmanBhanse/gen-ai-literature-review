# papers/sources/arxiv.py
"""arXiv paper source implementation."""

import time
from papers.sources.base import BasePaperSource, Paper
from papers.constants import MIN_REQUEST_DELAY

try:
    import arxiv
    ARXIV_AVAILABLE = True
except ImportError:
    ARXIV_AVAILABLE = False


class ArxivSource(BasePaperSource):
    """Fetch papers from arXiv (reliable for AI/ML papers, no rate limits)."""
    
    def fetch(self, query: str, num_papers: int = 1) -> list:
        """
        Fetch papers from arXiv.
        Simplifies queries to avoid rate limiting with long search terms.
        
        Args:
            query: Search query
            num_papers: Number of papers to fetch
            
        Returns:
            list[Paper]: List of papers or empty list if arxiv not available
        """
        if not ARXIV_AVAILABLE:
            self.logger.debug("arxiv not installed")
            return []
        
        papers = []
        try:
            # Simplify query to avoid rate limiting on long queries
            simplified_query = self._simplify_query(query)
            self.logger.info(f"Searching arXiv for: '{simplified_query}'")
            
            # Add delay to respect rate limits
            time.sleep(MIN_REQUEST_DELAY)
            
            client = arxiv.Client()
            search = arxiv.Search(
                query=simplified_query,
                max_results=num_papers,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for paper in client.results(search):
                if len(papers) >= num_papers:
                    break
                
                papers.append(Paper(
                    title=paper.title,
                    summary=paper.summary,
                    link=paper.entry_id,
                    source="arxiv",
                    year=paper.published.year if paper.published else None
                ))
            
            if papers:
                self.logger.info(f"✓ Found {len(papers)} paper(s) via arXiv")
        except Exception as e:
            self.logger.debug(f"arXiv search failed: {str(e)[:100]}")
        
        return papers
