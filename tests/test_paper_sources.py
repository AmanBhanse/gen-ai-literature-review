#!/usr/bin/env python3
"""
Test script to verify which paper sources (APIs) are working.
Tests: Google Scholar, Semantic Scholar (REST API), arXiv
"""

import sys
import time
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from papers.sources.google_scholar import GoogleScholarSource
from papers.sources.semantic_scholar import SemanticScholarSource
from papers.sources.arxiv import ArxivSource

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_google_scholar():
    """Test Google Scholar paper fetching"""
    print_section("Testing Google Scholar")
    
    test_query = "generative AI"
    print(f"Query: '{test_query}'")
    print("Fetching papers...\n")
    
    try:
        source = GoogleScholarSource()
        papers = source.fetch(test_query, num_papers=2)
        
        if papers:
            print(f"✅ SUCCESS: Found {len(papers)} paper(s)\n")
            for i, paper in enumerate(papers, 1):
                print(f"Paper {i}:")
                print(f"  Title:   {paper.title}")
                print(f"  Link:    {paper.link}")
                print(f"  Source:  {paper.source}")
                print(f"  Summary: {paper.summary[:100]}...\n")
            return True
        else:
            print("❌ FAILED: No papers found")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)[:200]}")
        return False

def test_semantic_scholar():
    """Test Semantic Scholar REST API"""
    print_section("Testing Semantic Scholar REST API")
    
    test_query = "machine learning"
    print(f"Query: '{test_query}'")
    print("Fetching papers...\n")
    
    try:
        source = SemanticScholarSource()
        papers = source.fetch(test_query, num_papers=2)
        
        if papers:
            print(f"✅ SUCCESS: Found {len(papers)} paper(s)\n")
            for i, paper in enumerate(papers, 1):
                print(f"Paper {i}:")
                print(f"  Title:   {paper.title}")
                print(f"  Link:    {paper.link}")
                print(f"  Source:  {paper.source}")
                print(f"  Summary: {paper.summary[:100]}...\n")
            return True
        else:
            print("❌ FAILED: No papers found")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)[:200]}")
        return False

def test_arxiv():
    """Test arXiv paper fetching"""
    print_section("Testing arXiv")
    
    test_query = "deep learning"
    print(f"Query: '{test_query}'")
    print("Fetching papers...\n")
    
    try:
        source = ArxivSource()
        papers = source.fetch(test_query, num_papers=2)
        
        if papers:
            print(f"✅ SUCCESS: Found {len(papers)} paper(s)\n")
            for i, paper in enumerate(papers, 1):
                print(f"Paper {i}:")
                print(f"  Title:   {paper.title}")
                print(f"  Link:    {paper.link}")
                print(f"  Source:  {paper.source}")
                print(f"  Summary: {paper.summary[:100]}...\n")
            return True
        else:
            print("❌ FAILED: No papers found")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)[:200]}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("   PAPER SOURCE AVAILABILITY TEST")
    print("="*60)
    
    results = {}
    
    print("\nTesting paper sources in sequence...")
    print("(This may take a minute as each test waits for API responses)\n")
    
    # Test each source
    print("1. Testing Google Scholar...")
    results['Google Scholar'] = test_google_scholar()
    time.sleep(2)
    
    print("\n2. Testing Semantic Scholar REST API...")
    results['Semantic Scholar'] = test_semantic_scholar()
    time.sleep(2)
    
    print("\n3. Testing arXiv...")
    results['arXiv'] = test_arxiv()
    
    # Summary
    print_section("TEST SUMMARY")
    
    for source, status in results.items():
        status_str = "✅ WORKING" if status else "❌ FAILED/RATE-LIMITED"
        print(f"{source:25} {status_str}")
    
    working_count = sum(1 for v in results.values() if v)
    print(f"\nTotal working sources: {working_count}/3")
    
    # Recommendations
    print_section("RECOMMENDATIONS")
    
    if results['Google Scholar']:
        print("✅ Google Scholar is working - using as primary source")
    else:
        print("⚠️  Google Scholar is not working")
    
    if results['arXiv']:
        print("✅ arXiv is working - can use as fallback")
    else:
        print("❌ arXiv is rate-limited or not working")
    
    if results['Semantic Scholar']:
        print("✅ Semantic Scholar is working - can use as fallback")
    else:
        print("❌ Semantic Scholar is rate-limited or not working")
    
    print("\nCurrent configuration in fetch_google_scholar_papers():")
    print("  1. Check cache first (fast, no API calls)")
    print("  2. Use Google Scholar (most reliable)")
    print("  3. Return papers or empty list")
    
    return 0 if working_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
