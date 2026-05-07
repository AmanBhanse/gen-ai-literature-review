# Gen-AI Literature Review System

This project automates the process of generating a literature review using a multi-agent system powered by LLMs (Large Language Models). It fetches papers, summarizes them, filters for relevance, creates a draft, and compares the AI-generated review with a human-written one using ROUGE metrics.

## Project Structure

```
gen-ai-literature-review/
├── main.py                          # Entry point
├── llm_client.py                    # LLM client (Groq/Llama3)
├── requirements.txt                 # Dependencies
├── output.txt                       # Generated literature review
│
├── agents/                          # Multi-agent system
│   ├── summarization_agent.py       # Paper summarization
│   ├── filter_agent.py              # Relevance filtering
│   ├── literature_review_writer.py  # Review generation
│   └── literature_review_editor.py  # Review refinement
│
├── workflows/                       # Orchestration layer
│   ├── orchestrator.py              # Main workflow coordinator
│   ├── summarization.py             # Summarization pipeline
│   ├── filter_papers.py             # Filtering pipeline
│   ├── literature_review_creation.py # Review creation pipeline
│   └── prompts.py                   # Centralized prompt management
│
├── papers/                          # Paper fetching & caching
│   ├── fetcher.py                   # Main paper fetching API
│   ├── cache.py                     # Local cache management
│   ├── constants.py                 # Paper-related constants
│   └── sources/                     # Multiple paper source integrations
│       ├── base.py                  # Abstract base source
│       ├── google_scholar.py        # Google Scholar (primary)
│       ├── semantic_scholar.py      # Semantic Scholar
│       ├── arxiv.py                 # arXiv
│       └── openalex.py              # OpenAlex
│
├── config_module/                   # Configuration management
│   ├── settings.py                  # Runtime settings
│   └── constants.py                 # Application constants
│
├── cli/                             # Command-line tools
│   └── precheck.py                  # Configuration validation
│
├── utils/                           # Shared utilities
│   ├── extraction.py                # Text extraction utilities
│   ├── metrics.py                   # ROUGE metrics
│   └── query.py                     # Query processing
│
├── docs/                            # Documentation
│   └── ARCHITECTURE.md              # Detailed architecture
│
└── tests/                           # Test suite
    ├── test_arxiv.py
    ├── test_google_scholar.py
    ├── test_openalex.py
    ├── test_semantic_scholar.py
    └── test_paper_sources.py
```

## Setup Instructions

1. **Clone the repository**

```sh
git clone <your-repo-url>
cd gen-ai-literature-review
```

2. **(Recommended) Create a virtual environment**

```sh
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```sh
pip install -r requirements.txt
```

4. **Configure API Keys**

Create a `.env` file in the project root directory with your Groq API key:

```sh
cp .env.example .env
```

Then edit `.env` and add your actual API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

⚠️ **Important**: Never commit the `.env` file to version control. It is ignored by `.gitignore` by default.

5. **Run the Project**

```sh
python main.py
```

This runs the full workflow to generate a literature review on your specified topic.

## Deactivating the Virtual Environment

To exit the virtual environment and return to your system Python:

- **macOS/Linux:**
  ```sh
  deactivate
  ```
- **Windows:**
  ```bat
  .\venv\Scripts\deactivate.bat
  ```

## Architecture Overview

**Modular Layered Design:**

- **CLI Layer**: Configuration validation (`cli/precheck.py`)
- **Workflow Layer**: Multi-step orchestration (`workflows/`)
- **Agent Layer**: Specialized LLM-powered agents (`agents/`)
- **Data Layer**: Paper fetching, caching, and utilities
- **LLM Client**: Unified interface to Groq/Llama3

For detailed architecture documentation, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Important Notes

### Paper Fetching & Caching

- **Multi-Source Strategy**: The system uses a hybrid approach with intelligent fallbacks
  - **Primary**: Google Scholar (most reliable, no strict rate limits)
  - **Fallbacks**: Semantic Scholar, arXiv, OpenAlex (all with rate limit handling)
  - **Caching**: Papers are cached locally in `paper_cache/` to avoid redundant API calls
- **Query Simplification**: Long queries (like full paper titles with author names) are automatically simplified
  - Extracts key terms to avoid API rate limiting and timeouts
  - Example: `"Advancements in Generative AI... by Author Name"` → `"Advancements Generative AI"`

### Configuration & Debugging

- **Environment Variables** (in `.env`):
  - `GROQ_API_KEY` - Your Groq API key (required)
  - `DEBUG_MODE` - Set to `true` for detailed logging (optional)

### Modular Architecture Benefits

This refactored structure provides:

- **Easy to Test**: Each module has a single responsibility
- **Easy to Extend**: Add new paper sources by creating a new class in `papers/sources/`
- **Easy to Maintain**: Clear separation of concerns
- **Easy to Reuse**: Import specific workflows or agents in your own projects

## Extending the System

### Adding a New Paper Source

1. Create a new file in `papers/sources/` extending `BaseSource`
2. Implement required methods: `fetch()`, `get_paper_metadata()`
3. Update the fetcher logic in `papers/fetcher.py`

### Creating Custom Workflows

Create new workflow files in `workflows/` that import and compose existing components:

```python
from workflows.summarization import summarize_papers_workflow
from agents.filter_agent import FilterAgent

async def custom_workflow():
    # Your custom orchestration here
    pass
```

## Troubleshooting

| Issue                   | Solution                                                      |
| ----------------------- | ------------------------------------------------------------- |
| Missing package errors  | Run `pip install -r requirements.txt`                         |
| API errors (e.g., Groq) | Verify `.env` file contains valid `GROQ_API_KEY`              |
| Paper fetching fails    | Check internet connection, verify API keys, check rate limits |
| Import errors           | Ensure you're running from the project root directory         |

## License

See `LICENSE` file for details.
