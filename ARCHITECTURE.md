# Module Architecture & Dependencies

## After Refactoring: New Module Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ENTRY POINT                                  │
│                       main.py (20 lines)                            │
│                   • Parse args  • Run workflow                      │
└────────────────────┬────────────────────────────────────────────────┘
                     │
        ┌────────────┴──────────────┐
        │                           │
        ▼                           ▼
┌──────────────────────┐    ┌──────────────────────┐
│   cli/precheck.py    │    │workflows/orchestrator│
│    (validation)      │    │   (main workflow)    │
└──────────────────────┘    └──────────┬───────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
          ▼                            ▼                            ▼
    ┌──────────────┐          ┌──────────────────┐      ┌─────────────────┐
    │ Summarization│          │ Filter Papers    │      │ Lit Review      │
    │  Workflow    │          │   Workflow       │      │   Workflow      │
    └──────┬───────┘          └────────┬─────────┘      └────────┬────────┘
           │                           │                         │
           └───────────────┬───────────┴─────────────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │workflows/prompts.py  │
                │ (Centralized messages)
                └──────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                       AGENTS LAYER                                  │
│  ┌────────────────┬────────────────┬────────────────────────────┐  │
│  │  Summarizer    │  Filter Agent  │  Lit Review Writer/Editor │  │
│  │    Agent       │     Agent      │       Agents              │  │
│  └────────────────┴────────────────┴────────────────────────────┘  │
│                          ▲                                          │
│                          │                                          │
│                  ┌───────┴────────┐                                 │
│                  │  llm_client.py │                                 │
│                  │ (Groq/Llama3)  │                                 │
│                  └────────────────┘                                 │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                     DATA LAYER                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │ papers/      │    │ config/      │    │ data/        │          │
│  │ (fetching)   │    │ (settings)   │    │ (models)     │          │
│  └──┬───────────┘    └──────────────┘    └──────────────┘          │
│     │                                                               │
│  ┌──┴────────────────────────────────────────────────────────────┐ │
│  │ • papers/sources/ - Individual paper sources                 │ │
│  │ • papers/cache.py - Cache management                         │ │
│  │ • papers/fetcher.py - Main routing API                       │ │
│  │ • utils/extraction.py - Content extraction                   │ │
│  │ • utils/query.py - Query simplification                      │ │
│  └──────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Dependency Graph (NEW STRUCTURE)

```
CURRENT                               AFTER REFACTORING

main.py                               main.py
  │                                     │
  ├─→ config.py                         ├─→ config/ ─→ settings.py
  ├─→ utils.py ⚠ (429 lines)            ├─→ papers/fetcher.py ──┐
  │     ├─ fetch_papers*()              │   ├─ papers/cache.py  │
  │     ├─ cache logic                  │   ├─ papers/sources/* │
  │     ├─ extraction                   │   └─ config/constants │
  │     └─ query utils                  │                       │
  │                                     ├─→ workflows/          │
  ├─→ workflows/                        │   └─ prompts.py       │
  │     ├─ orchestrator.py              │                       │
  │     ├─ summarization.py             ├─→ utils/extraction.py │
  │     ├─ filter_papers.py             ├─→ agents/             │
  │     ├─ lit_review_creation.py       └─→ cli/precheck.py     │
  │     └─ revising_draft.py
  │
  └─→ agents/
        ├─ summarization_agent.py
        ├─ filter_agent.py
        ├─ lit_review_writer.py
        └─ lit_review_editor.py


BEFORE: Star topology (everything imports utils.py)
┌─────┐
│ ... │←─┐
└─────┘  ├─ utils.py (center of universe)
┌─────┐  │
│ ... │←─┤
└─────┘  │
┌─────┐←──┘
│ ... │
└─────┘


AFTER: Clear layers (workflows → papers → config)
Papers Layer
  ├─ papers/sources/arxiv.py
  ├─ papers/sources/semantic_scholar.py
  ├─ papers/sources/openalex.py
  └─ papers/sources/google_scholar.py
       All ↓
    papers/fetcher.py
       ↓
    papers/cache.py
       ↓
     config/


Workflow Layer
  ├─ workflows/orchestrator.py
  ├─ workflows/summarization.py
  ├─ workflows/filter_papers.py
  └─ workflows/lit_review_creation.py
       All ↓
    workflows/prompts.py
       ↓
     config/


Utils Layer (Utilities - no cyclic dependencies)
  ├─ utils/extraction.py
  └─ utils/query.py
```

---

## Import Examples: Before & After

### BEFORE (Confusing)

```python
# In workflows/summarization.py
from utils import extract_draft_from_message, fetch_papers  # What is in utils?
from config import GROQ_API_KEY
from agents.summarization_agent import create_summarization_agent

# In workflows/filter_papers.py
from utils import fetch_papers  # Same Utils module, different functions
from config import PAPER_SOURCE

# In main.py
from utils import extract_draft_from_message
from config import GROQ_API_KEY
```

**Problem**: You have to know that `extract_draft_from_message` is in utils, `fetch_papers` is also in utils, etc.

### AFTER (Clear)

```python
# In workflows/summarization.py
from papers.fetcher import fetch_papers           # ✅ Clear: fetch from papers
from utils.extraction import ContentExtractor     # ✅ Clear: extract from utils
from config.settings import Settings              # ✅ Clear: settings from config
from agents.summarization_agent import create_summarization_agent

# In workflows/filter_papers.py
from papers.fetcher import fetch_papers           # ✅ Clear: same module
from config.constants import Constants            # ✅ Clear: constants from config

# In main.py
from utils.extraction import ContentExtractor     # ✅ Clear: extraction here
from config.settings import Settings              # ✅ Clear: settings here
```

**Benefit**: Module name tells you what's inside. `papers.fetcher.fetch_papers` is self-documenting.

---

## File Size Distribution: Before & After

### BEFORE (Unbalanced)

```
utils.py           ███████████████████████ 429 lines ⚠️ HUGE
config.py          ██ 33 lines
main.py            █████ 97 lines
metrics.py         █ 17 lines
llm_client.py      ██ 30 lines
────────────────────────────────
agents/* (each)    █ 13-14 lines (tiny, but 4 files)
workflows/* (avg)  ███ 60 lines
────────────────────────────────
Total LOC:         ~1030 lines in 19 files
Average: 54 lines per file
Problem: 42% of code in ONE file (utils.py)
```

### AFTER (Balanced)

```
papers/fetcher.py         ███ 30 lines
papers/cache.py           ███ 25 lines
papers/sources/arxiv.py   ████ 40 lines
papers/sources/s2.py      ████████ 58 lines
papers/sources/openalex.py ███████ 65 lines
papers/sources/gs.py      ████ 30 lines
papers/constants.py       ██ 15 lines
────────────────────────────────
config/settings.py        ██ 20 lines
config/constants.py       ██ 20 lines
config/api.py            ██ 15 lines
────────────────────────────────
workflows/prompts.py      ████ 40 lines
utils/extraction.py       ████ 35 lines
utils/query.py           ███ 20 lines
────────────────────────────────
cli/precheck.py          ████ 40 lines
────────────────────────────────
Total LOC:         ~1100 lines (similar)
Average: 30 lines per file
Distribution: Balanced across modules
Max file: 65 lines (still digestible)
```

---

## Module Responsibilities (After Refactoring)

```
┌─────────────────────────────────────────────────────────────┐
│ papers/                                                     │
│ ═══════                                                     │
│ Responsibility: Fetch papers from multiple sources         │
│ Does:     Source-specific API calls, caching               │
│ Does NOT: Filter, summarize, create reviews                │
│                                                             │
│ Public API:                                                 │
│   • papers.fetch_papers(query, num_papers)                 │
│   • papers.PaperFetcher.fetch()                            │
│                                                             │
├─ sources/arxiv.py          → ArxivSource                    │
├─ sources/semantic_scholar.py → SemanticScholarSource       │
├─ sources/openalex.py       → OpenAlexSource                │
├─ sources/google_scholar.py → GoogleScholarSource           │
├─ cache.py                  → CacheManager                  │
└─ fetcher.py                → PaperFetcher (router)         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ workflows/                                                  │
│ ═══════════                                                 │
│ Responsibility: Orchestrate multi-agent workflows           │
│ Does:     Coordinate agents, manage conversations           │
│ Does NOT: Fetch papers directly (uses papers module)        │
│                                                             │
│ Public API:                                                 │
│   • orchestrator.literature_review_generator_workflow()     │
│   • summarization.summarization_workflow()                  │
│   • prompts.PromptGenerator                                │
│                                                             │
├─ orchestrator.py           → Main workflow                  │
├─ summarization.py          → Summarization pipeline        │
├─ filter_papers.py          → Filtering pipeline            │
├─ literature_review_creation.py → Creation pipeline         │
├─ revising_draft.py         → Revision pipeline             │
└─ prompts.py                → PromptGenerator (centralized)  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ config/                                                     │
│ ════════                                                    │
│ Responsibility: Environment & static configuration          │
│ Does:     Load env vars, define constants, validate config │
│ Does NOT: Business logic                                    │
│                                                             │
│ Public API:                                                 │
│   • config.Settings                                        │
│   • config.Constants                                       │
│                                                             │
├─ settings.py               → Settings (from .env)          │
├─ constants.py              → Constants (static)            │
└─ api.py                    → API configurations            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ utils/                                                      │
│ ═══════                                                     │
│ Responsibility: Shared utilities (extraction, query tools)  │
│ Does:     Helper functions, not tied to one domain         │
│ Does NOT: Business logic, workflow orchestration            │
│                                                             │
│ Public API:                                                 │
│   • utils.extraction.ContentExtractor                      │
│   • utils.query.simplify_query()                           │
│                                                             │
├─ extraction.py             → ContentExtractor              │
├─ query.py                  → Query utilities               │
└─ logging.py                → Logging setup                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ agents/                                                     │
│ ════════                                                    │
│ Responsibility: Agent factory functions                     │
│ Does:     Create & configure agents                         │
│ Does NOT: Change (works well as-is)                         │
│                                                             │
│ Public API:                                                 │
│   • agents.create_summarization_agent()                    │
│   • agents.create_filter_agent()                           │
│   • agents.create_literature_review_writer_agent()         │
│   • agents.create_literature_review_editor_agent()         │
│                                                             │
├─ summarization_agent.py                                    │
├─ filter_agent.py                                           │
├─ literature_review_writer.py                               │
└─ literature_review_editor.py                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ cli/                                                        │
│ ════                                                        │
│ Responsibility: Entry point & validation                    │
│ Does:     Precheck, argument parsing                        │
│ Does NOT: Business logic                                    │
│                                                             │
│ Public API:                                                 │
│   • cli.precheck.Validator                                 │
│                                                             │
└─ precheck.py               → Validator class               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ data/                                                       │
│ ══════                                                      │
│ Responsibility: Data models & schemas                       │
│ Does:     Define types, validate data                       │
│ Does NOT: Fetch or process data                             │
│                                                             │
│ Public API:                                                 │
│   • data.Paper (Pydantic model)                            │
│   • data.Review (Pydantic model)                           │
│   • data.APIResponse (Pydantic model)                      │
│                                                             │
├─ models.py                 → Pydantic models               │
└─ schema.py                 → Type schemas                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Before & After

### BEFORE (Unclear data flow)

```
main.py
    ↓
    uses: utils.fetch_papers()
    ↓
    returns: list[dict] ← What keys? What format?
    ↓
    passed to: workflows/summarization.py
    ↓
    uses: utils.extract_draft_from_message()
    ↓
    returns: str | None ← Sometimes None? When?
    ↓
    passed to: workflows/filter_papers.py
```

**Problem**: No type hints on data. Hard to know what you're getting.

### AFTER (Clear data flow with types)

```
main.py
    ↓
    uses: papers.fetch_papers(query: str) → list[Paper]
    ↓
    returns: list[Paper] ← Pydantic model with validation
           .title: str
           .summary: str
           .link: str
           .source: str
    ↓
    passed to: workflows/summarization.py
             (receives: list[Paper])
    ↓
    uses: ContentExtractor.extract(msg) → Optional[str]
    ↓
    returns: str | None ← Type hint says when None
    ↓
    passed to: workflows/filter_papers.py
             (receives: Optional[str])
```

**Benefit**: Every module documents what it expects & returns.

---

## Adding a New Paper Source: Before vs After

### BEFORE (Modify existing code)

```python
# In utils.py, add new function:
def fetch_papers_new_source(query, num_papers=1):
    # 40+ lines of new logic
    ...

# In config.py, add to PAPER_SOURCE options:
# "new_source"  ← manually add to comment

# In main.py or precheck(), add validation:
if PAPER_SOURCE == "new_source":
    # handle new source

# In tests/test_paper_sources.py, add test:
def test_new_source():
    # new test
```

**Problem**: Changes scattered across 5+ files. Easy to forget things.

### AFTER (Add new file)

```python
# Create: papers/sources/new_source.py
from papers.sources.base import BasePaperSource, Paper

class NewSource(BasePaperSource):
    def fetch(self, query: str, num_papers: int = 1) -> list[Paper]:
        """Fetch from new source"""
        # Your 40 lines of logic here
        ...

# Update: papers/fetcher.py (1 line added)
self.sources["new_source"] = NewSource()

# Create: tests/sources/test_new_source.py
def test_new_source():
    source = NewSource()
    papers = source.fetch("ai", 3)
    assert len(papers) <= 3
    assert isinstance(papers[0], Paper)
```

**Benefit**: New source is isolated. No touch to existing code. Plugin architecture!

---

## Summary: Why This Refactoring Matters

| Aspect                 | Benefit                                                      |
| ---------------------- | ------------------------------------------------------------ |
| **Readability**        | Import paths document what you're getting (`papers.fetcher`) |
| **Maintainability**    | Changes localized to one module                              |
| **Testability**        | Each module independently testable with mocks                |
| **Extensibility**      | Add new features without modifying existing code             |
| **Onboarding**         | New devs understand structure from folder names              |
| **Debugging**          | Stack traces point to specific modules, not generic "utils"  |
| **Refactoring Safety** | Small focused files are easier to refactor                   |
