# Gen-AI Literature Review System

This project automates the process of generating a literature review using a multi-agent system powered by LLMs (Large Language Models). It fetches papers, summarizes them, filters for relevance, creates a draft, and compares the AI-generated review with a human-written one using ROUGE metrics.

## Project Structure

```
├── agents.py           # Agent and client construction
├── config.py           # API keys and global constants
├── main.ipynb          # Jupyter notebook for experimentation
├── main.py             # Main script to run the workflow
├── metrics.py          # ROUGE and other evaluation metrics
├── requirements.txt    # Python dependencies
├── utils.py            # Utility functions
├── workflows.py        # All async workflows
├── .env                # Environment variables (do not commit)
├── .env.example        # Template for environment variables
├── .gitignore          # Files and folders to ignore in git
└── README.md           # This file
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

- To run the full workflow and see results:

```sh
python main.py
```

- Or, use `main.ipynb` for step-by-step experimentation in Jupyter Notebook.

## Deactivating the Virtual Environment

- **macOS/Linux:**
  ```sh
  deactivate
  ```
- **Windows:**
  ```bat
  .\venv\Scripts\deactivate.bat
  ```

This will return you to your system's default Python environment.

## Notes

- **Paper Search**: The system uses a hybrid approach with caching to fetch papers:
  - **Primary**: Google Scholar (most reliable, no strict rate limits)
  - **Caching**: Papers are cached locally in `paper_cache/` to avoid repeated API calls and speed up searches
  - ⚠️ Note: Semantic Scholar and arXiv both have strict rate limits on free tier; Google Scholar is more reliable
- **Paper Search Query Simplification**:
  - Long queries (like full paper titles with author names) are automatically simplified
  - Extracts key terms to avoid API rate limiting and timeouts
  - Example: "Advancements in Generative AI... by Author Name" → "Advancements Generative Comprehensive"
- **Debug Logging** (in `.env`):
  - `DEBUG_MODE=true` - Shows detailed information about paper fetching and processing
  - `DEBUG_MODE=false` (default) - Only warnings and errors displayed

- All modular code is in `.py` files for easy reuse and testing.
- Results are compared to a human-written review using ROUGE metrics.

## Troubleshooting

- If you see missing package errors, ensure you have installed all dependencies from `requirements.txt`.
- For API errors, check that your `.env` file is created and contains a valid `GROQ_API_KEY`.
- If you see `__pycache__` or other unnecessary files, they are ignored by `.gitignore`.

## License

See `LICENSE` file for details.
