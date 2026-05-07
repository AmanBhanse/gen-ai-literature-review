#!/usr/bin/env python3
"""
Run all paper source tests and generate a summary report
Useful for diagnosing which APIs are working or rate-limited
Run from: cd tests && python run_all_tests.py
"""

import subprocess
import sys
import time
from pathlib import Path

def run_test(test_file, source_name):
    """Run a test file and return True if successful"""
    print(f"\n{'='*60}")
    print(f"Running: {test_file}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=False,
            text=True,
            timeout=30,
            cwd=Path(__file__).parent
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"❌ Test timed out after 30 seconds")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("   COMPREHENSIVE PAPER SOURCE TEST SUITE")
    print("="*60)
    print("\nThis will test all available paper sources.")
    print("Testing order: Google Scholar → Semantic Scholar → arXiv\n")
    
    tests = [
        ("test_google_scholar.py", "Google Scholar"),
        ("test_semantic_scholar.py", "Semantic Scholar REST API"),
        ("test_arxiv.py", "arXiv"),
    ]
    
    results = {}
    
    for test_file, source_name in tests:
        print(f"\n[{len(results)+1}/{len(tests)}] Testing {source_name}...")
        success = run_test(test_file, source_name)
        results[source_name] = success
        time.sleep(3)  # Wait between tests to avoid rate limiting
    
    # Print summary report
    print("\n\n" + "="*60)
    print("   TEST SUMMARY REPORT")
    print("="*60 + "\n")
    
    for source_name, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED/RATE-LIMITED"
        print(f"{source_name:30} {status}")
    
    working_count = sum(1 for v in results.values() if v)
    print(f"\nTotal working sources: {working_count}/{len(results)}")
    
    # Recommendations
    print("\n" + "="*60)
    print("   RECOMMENDATIONS")
    print("="*60 + "\n")
    
    if results.get("Google Scholar"):
        print("✅ Google Scholar is available - using as PRIMARY source")
    else:
        print("⚠️  Google Scholar is not working - may need to run from different IP")
    
    working_sources = [s for s, success in results.items() if success]
    if len(working_sources) > 1:
        print(f"\n✅ Multiple sources available: {', '.join(working_sources[1:])}")
    elif len(working_sources) == 0:
        print("\n❌ WARNING: No paper sources are working!")
        print("   This could be due to:")
        print("   - Network connectivity issues")
        print("   - All APIs are rate-limiting this IP")
        print("   - Dependencies not installed correctly")
    
    print("\nCurrent paper search logic:")
    print("  1. Check paper_cache/ for previously fetched papers (instant)")
    print("  2. Use Google Scholar for new paper searches")
    print("  3. Cache results for future use\n")
    
    return 0 if working_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
