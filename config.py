# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable is not set. "
        "Please create a .env file in the project root with: GROQ_API_KEY=your_api_key"
    )

# Output Separators
OUTPUT_SEPERATOR_START = "-------------- OUTPUT : STARTS ----------------"
OUTPUT_SEPERATOR_END = "-------------- OUTPUT : ENDS ----------------"

# Word Counts
LITERATURE_REVIEW_WORD_COUNT = 500
SINGLE_PAPER_SUMMARY_WORD_COUNT = 100
