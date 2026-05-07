#!/usr/bin/env python3
"""
Test Google Scholar paper source independently
"""

import sys
from pathlib import Path

# Add parent directory to path to import scholarly
sys.path.insert(0, str(Path(__file__).parent.parent))

from scholarly import scholarly

def test_google_scholar():
    """Test Google Scholar directly"""
    print("\n" + "="*60)
    print("  Testing Google Scholar API")
    print("="*60 + "\n")
    
    query = "generative AI machine learning"
    print(f"Query: '{query}'")
    print("Fetching 3 papers...\n")
    
    try:
        print("Searching scholarly.search_pubs()...")
        search_results = scholarly.search_pubs(query)
        
        papers_found = []
        for i, paper in enumerate(search_results):
            if i >= 3:
                break
            
            papers_found.append({
                "title": paper["bib"]["title"],
                "summary": paper["bib"].get("abstract", "No abstract"),
                "link": paper.get("pub_url", "No URL"),
                "source": "google_scholar"
            })
        
        if papers_found:
            print(f"✅ SUCCESS: Found {len(papers_found)} papers\n")
            for i, paper in enumerate(papers_found, 1):
                print(f"Paper {i}:")
                print(f"  Title:   {paper['title']}")
                print(f"  Link:    {paper['link'][:60]}...")
                print(f"  Summary: {paper['summary'][:80]}...\n")
            return True
        else:
            print("❌ FAILED: No papers returned")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}")
        print(f"Message: {str(e)[:300]}")
        return False

if __name__ == "__main__":
    success = test_google_scholar()
    
    print("="*60)
    if success:
        print("✅ Google Scholar is WORKING")
    else:
        print("❌ Google Scholar FAILED")
    print("="*60 + "\n")
