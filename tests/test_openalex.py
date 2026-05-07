#!/usr/bin/env python3
"""
Test OpenAlex API independently
Requires API key from https://openalex.org/settings/api
"""

import os
import sys
import requests
from pathlib import Path

# Load .env file from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

OPENALEX_BASE_URL = "https://api.openalex.org"
OPENALEX_WORKS_ENDPOINT = f"{OPENALEX_BASE_URL}/works"

def test_openalex():
    """Test OpenAlex API directly"""
    print("\n" + "="*60)
    print("  Testing OpenAlex API")
    print("="*60 + "\n")
    
    # Get API key from environment
    api_key = os.getenv("OPENALEX_API_KEY")
    
    if not api_key:
        print("WARNING: OPENALEX_API_KEY not set in environment")
        print("\nTo get your free API key:")
        print("  1. Visit: https://openalex.org/settings/api")
        print("  2. Sign up or log in")
        print("  3. Add to .env file: OPENALEX_API_KEY=your_key_here")
        print("\nOpenAlex API is FREE - no credit card required!\n")
        return False
    
    query = "generative AI machine learning"
    print(f"Query: '{query}'")
    print("Fetching 5 papers from OpenAlex...\n")
    
    try:
        print("Making request to OpenAlex API...")
        params = {
            "search": query,
            "per_page": 5,
            "api_key": api_key
        }
        
        response = requests.get(OPENALEX_WORKS_ENDPOINT, params=params, timeout=10)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            meta = data.get("meta", {})
            
            print(f"Total matches: {meta.get('count', 'N/A')}\n")
            
            if results:
                print(f"✅ SUCCESS: Found {len(results)} papers (showing first 5)\n")
                for i, paper in enumerate(results, 1):
                    print(f"Paper {i}:")
                    print(f"  Title:         {paper.get('title', 'N/A')[:70]}...")
                    print(f"  OpenAlex ID:   {paper.get('id', 'N/A')}")
                    print(f"  Publication:   {paper.get('publication_year', 'N/A')}")
                    print(f"  Citation Count: {paper.get('cited_by_count', 'N/A')}")
                    
                    # Get first author if available
                    authorships = paper.get('authorships', [])
                    if authorships:
                        first_author = authorships[0].get('author', {}).get('display_name', 'N/A')
                        print(f"  First Author:  {first_author}")
                    
                    print()
                return True
            else:
                print("❌ FAILED: No papers in response")
                return False
                
        elif response.status_code == 401:
            print(f"❌ AUTHENTICATION FAILED (HTTP 401)")
            print("Invalid API key. Check your OPENALEX_API_KEY")
            return False
        elif response.status_code == 429:
            print(f"❌ RATE LIMITED (HTTP 429)")
            print("OpenAlex is rate limiting this request")
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
    success = test_openalex()
    
    print("="*60)
    if success:
        print("✅ OpenAlex API is WORKING")
    else:
        print("❌ OpenAlex API FAILED or NOT CONFIGURED")
    print("="*60 + "\n")
