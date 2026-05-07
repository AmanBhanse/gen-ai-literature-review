# config_module/settings.py
"""Environment-based configuration settings."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Configurable settings loaded from environment."""
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", None)
    SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", None)
    
    # Logging Configuration
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
    
    # Paper Search Configuration
    PAPER_SOURCE = os.getenv("PAPER_SOURCE", "google_scholar")
    
    # Caching
    ENABLE_PAPER_CACHE = os.getenv("ENABLE_PAPER_CACHE", "true").lower() == "true"
    
    @classmethod
    def validate(cls):
        """Validate required settings on startup."""
        if not cls.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY environment variable is not set. "
                "Please create a .env file in the project root with: GROQ_API_KEY=your_api_key"
            )
    
    @classmethod
    def as_dict(cls) -> dict:
        """Get all settings as a dictionary."""
        return {
            "GROQ_API_KEY": cls.GROQ_API_KEY[:10] + "***" if cls.GROQ_API_KEY else None,  # mask for security
            "OPENALEX_API_KEY": "***" if cls.OPENALEX_API_KEY else None,
            "DEBUG_MODE": cls.DEBUG_MODE,
            "PAPER_SOURCE": cls.PAPER_SOURCE,
            "ENABLE_PAPER_CACHE": cls.ENABLE_PAPER_CACHE,
        }
