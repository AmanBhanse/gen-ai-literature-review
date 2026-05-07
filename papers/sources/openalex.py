# papers/sources/openalex.py
"""OpenAlex paper source implementation."""

import time
import requests
from papers.sources.base import BasePaperSource, Paper
from papers.constants import OPENALEX_API_BASE, MIN_REQUEST_DELAY


class OpenAlexSource(BasePaperSource):
    """Fetch papers from OpenAlex (comprehensive, no strict rate limits)."""
    
    def __init__(self, api_key: str = None):
        super().__init__()
        self.api_key = api_key
    
    def fetch(self, query: str, num_papers: int = 1) -> list:
        """
        Fetch papers from OpenAlex.
        Requires OPENALEX_API_KEY in .env file.
        
        Args:
            query: Search query
            num_papers: Number of papers to fetch
            
        Returns:
            list[Paper]: List of papers or empty list on failure
        """
        if not self.api_key:
            self.logger.error("OPENALEX_API_KEY not set. Get a free key at https://openalex.org/settings/api")
            return []
        
        papers = []
        try:
            simplified_query = self._simplify_query(query)
            self.logger.info(f"Searching OpenAlex for: '{simplified_query}'")
            
            # Add delay to respect rate limits
            time.sleep(MIN_REQUEST_DELAY)
            
            params = {
                "search": simplified_query,
                "per_page": num_papers,
                "mailto": "user@example.com"  # OpenAlex requires this
            }
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = requests.get(OPENALEX_API_BASE, params=params, headers=headers, timeout=10)
            
            if response.status_code == 401:
                self.logger.error("OpenAlex authentication failed - invalid API key")
                return []
            elif response.status_code == 429:
                self.logger.warning("OpenAlex rate limited")
                return []
            elif response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                if not results:
                    self.logger.debug("OpenAlex returned no results")
                    return []
                
                for work in results:
                    if work.get("title"):
                        authors = ", ".join([a["author"]["display_name"] for a in work.get("authorships", [])[:3]])
                        papers.append(Paper(
                            title=work.get("title", "No title"),
                            summary=work.get("abstract") or work.get("title", "No abstract available"),
                            link=work.get("id", "No link"),
                            source="openalex",
                            year=work.get("publication_year"),
                            citations=work.get("cited_by_count", 0),
                            authors=authors
                        ))
                
                if papers:
                    self.logger.info(f"✓ Found {len(papers)} paper(s) via OpenAlex")
                return papers
            else:
                self.logger.debug(f"OpenAlex API returned {response.status_code}")
                return []
                
        except requests.Timeout:
            self.logger.debug("OpenAlex API request timed out")
            return []
        except Exception as e:
            self.logger.debug(f"OpenAlex API error: {type(e).__name__}: {str(e)[:100]}")
            return []
