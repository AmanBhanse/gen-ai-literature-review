#!/usr/bin/env python3
"""
Test arXiv API independently
"""

import arxiv

def test_arxiv():
    """Test arXiv API directly"""
    print("\n" + "="*60)
    print("  Testing arXiv API")
    print("="*60 + "\n")
    
    query = "vision transformer"
    print(f"Query: '{query}'")
    print("Fetching 3 papers...\n")
    
    try:
        print("Connecting to arXiv API...")
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        print("Processing results...\n")
        results = []
        for paper in client.results(search):
            results.append(paper)
        
        if results:
            print(f"✅ SUCCESS: Found {len(results)} papers\n")
            for i, paper in enumerate(results, 1):
                print(f"Paper {i}:")
                print(f"  Title:     {paper.title}")
                print(f"  Authors:   {', '.join([a.name for a in paper.authors[:3]])}...")
                print(f"  Published: {paper.published.date()}")
                print(f"  Link:      {paper.entry_id}\n")
            return True
        else:
            print("❌ FAILED: No papers found")
            return False
            
    except arxiv.HTTPError as e:
        if "429" in str(e):
            print(f"❌ RATE LIMITED (HTTP 429)")
            print("arXiv is rate limiting this IP")
        else:
            print(f"❌ HTTP ERROR: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}")
        print(f"Message: {str(e)[:300]}")
        return False

if __name__ == "__main__":
    success = test_arxiv()
    
    print("="*60)
    if success:
        print("✅ arXiv API is WORKING")
    else:
        print("❌ arXiv API FAILED or RATE-LIMITED")
    print("="*60 + "\n")
