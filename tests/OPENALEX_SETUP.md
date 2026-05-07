# OpenAlex Integration Guide

## What is OpenAlex?

**OpenAlex** is a free, open-source index of scholarly papers covering millions of works from all fields of science and scholarship.

- 📚 **Coverage**: All academic papers, preprints, books
- 🆓 **Cost**: Completely free
- 🔑 **API Key**: Free (no credit card required)
- 📊 **Data**: More comprehensive than arXiv alone
- ⚡ **Speed**: Very fast, no strict rate limits

## Getting Your Free API Key

### Step 1: Visit OpenAlex Settings

https://openalex.org/settings/api

### Step 2: Sign Up or Log In

- If you don't have an account, sign up with email
- It's completely free

### Step 3: Copy Your API Key

- Once logged in, your API key will be displayed
- It looks like: `YOUR_KEY_HERE` (exact format varies)

### Step 4: Add to Your .env File

Edit `.env` file in project root:

```env
GROQ_API_KEY=gsk_...
OPENALEX_API_KEY=your_api_key_here
DEBUG_MODE=false
PAPER_SOURCE=hybrid
ENABLE_PAPER_CACHE=true
```

## Testing OpenAlex

### Test if Your API Key Works

```bash
cd tests
python test_openalex.py
```

Expected output:

```
============================================================
  Testing OpenAlex API
============================================================

Query: 'generative AI machine learning'
Fetching 5 papers from OpenAlex...

Making request to OpenAlex API...
Response status: 200

Total matches: 12345

✅ SUCCESS: Found 5 papers (showing first 5)

Paper 1:
  Title:         Advancements in Generative AI: A Comprehensive...
  OpenAlex ID:   https://openalex.org/W1234567890
  Publication:   2024
  Citation Count: 45
  First Author:  John Smith

...

✅ OpenAlex API is WORKING
```

## Using OpenAlex in Your System

The system now uses **single-source paper fetching** (no fallback chain). Simply set your preferred source in the `.env` file.

### Use OpenAlex as Your Paper Source

**Option 1: Primary Source (Recommended for best coverage)**

Edit `.env`:

```env
GROQ_API_KEY=gsk_...
OPENALEX_API_KEY=your_api_key_from_above
PAPER_SOURCE=openalex
ENABLE_PAPER_CACHE=true
```

Then test:

```bash
python main.py
```

The system will fetch all papers from OpenAlex exclusively.

### Use Google Scholar (Default - No API Key Needed)

Edit `.env`:

```env
GROQ_API_KEY=gsk_...
PAPER_SOURCE=google_scholar
ENABLE_PAPER_CACHE=true
```

Google Scholar is the default source and requires no API key.

### Other Available Sources

You can also choose:

```env
PAPER_SOURCE=arxiv              # AI/ML papers only
PAPER_SOURCE=semantic_scholar   # Academic focus
```

### How the System Works

1. **Check Cache First** - Previously fetched papers (fastest)
2. **Use Selected Source** - Only the `PAPER_SOURCE` you configured
3. **No Fallback** - If the source fails, the system reports the failure
4. **Save to Cache** - Results are cached for future use

This approach ensures **predictable, transparent behavior** - you always know which source is being used.

## OpenAlex API Features

### Available Data Fields

Each paper from OpenAlex includes:

- `title` - Paper title
- `abstract` - Paper abstract (when available)
- `publication_year` - Year published
- `cited_by_count` - Number of citations
- `authorships` - List of authors
- `open_access` - Open access information
- `doi` - Digital Object Identifier
- `id` - OpenAlex persistent identifier

### Search Syntax

Examples:

```bash
# Simple search
# "generative AI"

# Title search
# "title: \"machine learning\""

# Author search
# "author: \"LeCun\""

# Year filter
# "publication_year: >2020"

# Combination
# "generative AI publication_year: 2023..2024"
```

## Comparison with Other Sources

| Source           | Requires API Key | Rate Limits     | Coverage    | Speed       |
| ---------------- | ---------------- | --------------- | ----------- | ----------- |
| Google Scholar   | No               | Yes (strict)    | Very High   | Slow        |
| Semantic Scholar | No (optional)    | Yes (~100/5min) | High        | Fast        |
| arXiv            | No               | Yes (~30/min)   | AI/ML only  | Medium      |
| **OpenAlex**     | **Yes (free)**   | **No**          | **Highest** | **Fastest** |

## Troubleshooting

### Test shows: "⚠️ WARNING: OPENALEX_API_KEY not set"

- Make sure `.env` file exists
- Add the line: `OPENALEX_API_KEY=your_key`
- Restart Python/IDE

### Test shows: "❌ AUTHENTICATION FAILED (HTTP 401)"

- API key is invalid or expired
- Get a new key from: https://openalex.org/settings/api

### Test shows: "❌ RATE LIMITED (HTTP 429)"

- Very unlikely with OpenAlex (no strict limits)
- Wait a few seconds and retry
- Check OpenAlex status page

### Papers found but with empty abstracts

- Not all papers in OpenAlex have abstracts
- This is normal - abstracts are optional

## Full Implementation

Complete code ready to use:

See `openalex_integration.py` in this folder for:

- Function signature
- Error handling
- Response parsing
- Example usage

## Next Steps

1. ✅ Get your free API key: https://openalex.org/settings/api
2. ✅ Add to `.env` file: `OPENALEX_API_KEY=your_key`
3. ✅ Set paper source: `PAPER_SOURCE=openalex` (or choose another source)
4. ✅ Run test: `python tests/test_openalex.py`
5. ✅ Test system: `python main.py`

The system is now ready to use with your chosen paper source!

## Questions?

- OpenAlex Docs: https://developers.openalex.org/
- Get API Key: https://openalex.org/settings/api
- GitHub: https://github.com/ourresearch/openalex
