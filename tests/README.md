# Test Files

This folder contains diagnostic tests for the paper sources used by the literature review system.

## Test Files

### Individual Source Tests

- **test_google_scholar.py** - Tests Google Scholar paper fetching (no API key required)
- **test_semantic_scholar.py** - Tests Semantic Scholar REST API (no API key required)
- **test_arxiv.py** - Tests arXiv paper fetching (no API key required)
- **test_openalex.py** - Tests OpenAlex API (requires free API key)

### Comprehensive Test Suite

- **test_paper_sources.py** - Tests all sources and imports from utils
- **run_all_tests.py** - Master test runner (runs all individual tests)

### Integration & Documentation

- **openalex_integration.py** - Ready-to-use OpenAlex integration code for utils.py
- **OPENALEX_SETUP.md** - Complete guide to set up OpenAlex (free API key)

## How to Run Tests

### Run All Tests at Once

```bash
cd tests
python run_all_tests.py
```

### Run Specific Test

```bash
cd tests
python test_google_scholar.py        # Test Google Scholar (no API key)
python test_semantic_scholar.py      # Test Semantic Scholar REST API (no API key)
python test_arxiv.py                 # Test arXiv (no API key)
python test_openalex.py              # Test OpenAlex (requires free API key)
```

### Run Integrated Test (from utils)

```bash
cd tests
python test_paper_sources.py
```

## What Each Test Checks

### Google Scholar Test

- Queries for "generative AI machine learning"
- Fetches 3 papers
- Checks for title, URL, abstract availability
- ✅ Indicates if Google Scholar is working

### Semantic Scholar Test

- Makes REST API request to Semantic Scholar
- Queries for "transformer neural network"
- Fetches 3 papers
- Detects rate limiting (HTTP 429)
- ✅ Indicates if SS API is working or rate-limited

### arXiv Test

- Queries for "vision transformer"
- Fetches 3 papers
- Shows paper details and links
- Detects rate limiting (HTTP 429)
- ✅ Indicates if arXiv is working or rate-limited

### OpenAlex Test

- Requires **free API key** from https://openalex.org/settings/api
- Queries for "generative AI machine learning"
- Fetches 5 papers with detailed metadata
- Shows: title, authors, year, citations, OpenAlex ID
- Checks for authentication (401), rate limiting (429)
- ✅ Indicates if OpenAlex is working and API key is valid

**Note**: First run will show warning if API key not configured - this is expected!

## Interpreting Results

### Expected Output

```
✅ WORKING       = API is available and returning results
❌ RATE-LIMITED  = API returned HTTP 429 (too many requests)
❌ FAILED        = API error or no results
```

### Known Issues

**arXiv Rate Limiting**: If running tests frequently, arXiv may rate limit your IP. Wait 30-60 seconds before retrying.

**Semantic Scholar Rate Limiting**: Free tier has strict limits (~100 req/5 min). System automatically delays requests 5 seconds between API calls.

**Google Scholar**: Most reliable but occasionally blocks on rate limits. If blocked, changing your IP (VPN) may help.

**OpenAlex**: No strict rate limits, but requires a free API key. See OPENALEX_SETUP.md for setup.

## Paper Source Selection

The system uses a **single paper source** configured in `.env` - **no fallback chain**.

Set `PAPER_SOURCE` in your `.env` file to one of:

- `google_scholar` (default) - Most reliable, no API key needed
- `arxiv` - Best for AI/ML papers, no API key needed
- `semantic_scholar` - More academic focus, no API key needed
- `openalex` - Most comprehensive, requires free API key

**Important**: Only the selected source will be used. If it fails, the system will report failure rather than falling back to another source.

### Source Comparison

| Source               | API Key | Rate Limit | Coverage   | Reliability |
| -------------------- | ------- | ---------- | ---------- | ----------- |
| **Google Scholar**   | No      | Occasional | Very Good  | Good        |
| **arXiv**            | No      | Yes (429)  | AI/ML only | Fair        |
| **Semantic Scholar** | No      | Yes (429)  | Academic   | Fair        |
| **OpenAlex**         | Free    | None       | Excellent  | Excellent   |

Use **OpenAlex** for best results if you can get the free API key. Otherwise, **Google Scholar** is the recommended default.

**Note**: To use OpenAlex, get your free API key from https://openalex.org/settings/api and add to .env

## Troubleshooting

### All tests failing

- Check internet connection
- Verify dependencies: `pip install -r requirements.txt`
- Your IP may be rate-limited by all services

### Only Google Scholar works

- arXiv and Semantic Scholar are rate-limited
- System will use Google Scholar as primary source ✅

### Only Semantic Scholar works

- Google Scholar may be blocking your IP
- Try again later or use a different IP/VPN

## Running Tests in CI/CD

These tests can be integrated into continuous integration:

```bash
python -m pytest tests/test_paper_sources.py -v
```

Or using unittest:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Performance Notes

- Each test takes ~10-20 seconds (API response time)
- Full test suite takes ~45-60 seconds
- Tests add 3-second delays between API calls to avoid rate limiting
- Results are not cached; each run makes fresh API calls
