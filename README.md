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

- The `GROQ_API_KEY` is stored in `config.py`. Replace the placeholder with your actual key if needed.

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

- The system uses Google Scholar for paper search and LLMs for summarization and review generation.
- All modular code is in `.py` files for easy reuse and testing.
- Results are compared to a human-written review using ROUGE metrics.

## Troubleshooting

- If you see missing package errors, ensure you have installed all dependencies from `requirements.txt`.
- For API errors, check your API key in `config.py`.
- If you see `__pycache__` or other unnecessary files, they are ignored by `.gitignore`.

## License

See `LICENSE` file for details.
