#!/usr/bin/env python3
"""
Test Semantic Scholar REST API independently
"""

import requests

S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_PAPER_SEARCH_ENDPOINT = f"{S2_API_BASE}/paper/search"
S2_FIELDS = "abstract,authors,citationCount,publicationDate,title,url,venue,year"

def test_semantic_scholar():
    """Test Semantic Scholar REST API directly"""
    print("\n" + "="*60)
    print("  Testing Semantic Scholar REST API")
    print("="*60 + "\n")
    
    query = "transformer neural network"
    print(f"Query: '{query}'")
    print("Fetching 3 papers...\n")
    
    try:
        print("Making request to Semantic Scholar REST API...")
        params = {
            "query": query,
            "limit": 3,
            "fields": S2_FIELDS
        }
        
        response = requests.get(S2_PAPER_SEARCH_ENDPOINT, params=params, timeout=10)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("data", [])
            
            if results:
                print(f"✅ SUCCESS: Found {len(results)} papers\n")
                for i, paper in enumerate(results, 1):
                    print(f"Paper {i}:")
                    print(f"  Title:  {paper.get('title', 'N/A')}")
                    print(f"  URL:    {paper.get('url', 'N/A')}")
                    print(f"  Year:   {paper.get('year', 'N/A')}")
                    print(f"  Venue:  {paper.get('venue', 'N/A')}\n")
                return True
            else:
                print("❌ FAILED: No papers in response")
                return False
        elif response.status_code == 429:
            print(f"❌ RATE LIMITED (HTTP 429)")
            print("Semantic Scholar is rate limiting this IP")
            return False
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
            
    except requests.Timeout:
        print("❌ ERROR: Request timed out (10s)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}")
        print(f"Message: {str(e)[:300]}")
        return False

if __name__ == "__main__":
    success = test_semantic_scholar()
    
    print("="*60)
    if success:
        print("✅ Semantic Scholar REST API is WORKING")
    else:
        print("❌ Semantic Scholar REST API FAILED or RATE-LIMITED")
    print("="*60 + "\n")
