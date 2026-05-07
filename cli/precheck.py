# cli/precheck.py
"""Validation and health checks for application startup."""

import sys
import logging
from config_module import Settings

logger = logging.getLogger(__name__)


class Validator:
    """Validates application configuration and dependencies."""
    
    @staticmethod
    def check_api_keys() -> bool:
        """Check that required API keys are configured."""
        if not Settings.GROQ_API_KEY or not isinstance(Settings.GROQ_API_KEY, str) or not Settings.GROQ_API_KEY.startswith("gsk_"):
            logger.error("GROQ_API_KEY is missing or invalid. Please check your .env file.")
            return False
        return True
    
    @staticmethod
    def check_groq_endpoint() -> bool:
        """Verify Groq endpoint is reachable and API key is valid."""
        try:
            import openai
            client = openai.OpenAI(
                api_key=Settings.GROQ_API_KEY,
                base_url="https://api.groq.com/openai/v1"
            )
            # List models as a lightweight test
            _ = client.models.list()
            logger.info("✓ GROQ API key is valid and Groq endpoint is reachable.")
            return True
        except Exception as e:
            logger.error(f"Failed to validate GROQ API key or reach Groq endpoint: {e}")
            return False
    
    @staticmethod
    def check_openalex_rate_limits() -> bool:
        """Check OpenAlex API rate limit status using the rate-limit endpoint."""
        try:
            import requests
            # Use the dedicated rate-limit endpoint with API key if available
            url = "https://api.openalex.org/rate-limit"
            params = {}
            if Settings.OPENALEX_API_KEY:
                params["api_key"] = Settings.OPENALEX_API_KEY
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                rate_limit = data.get("rate_limit", {})
                
                # Extract useful rate limit info
                daily_remaining_usd = rate_limit.get("daily_remaining_usd", 0)
                credits_remaining = rate_limit.get("credits_remaining", 0)
                resets_in = rate_limit.get("resets_in_seconds", 0)
                
                logger.info(f"✓ OpenAlex rate limits: ${daily_remaining_usd:.4f}/day remaining, {credits_remaining} credits remaining (resets in {resets_in}s)")
                return True
            else:
                logger.error(f"Failed to check OpenAlex rate limits: HTTP {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Failed to check OpenAlex rate limits: {e}")
            return False
    
    @classmethod
    def run_all_checks(cls) -> bool:
        """Run all validation checks."""
        print("\n[Pre-check] Verifying configuration and API access...")
        
        if not cls.check_api_keys():
            return False
        
        if not cls.check_groq_endpoint():
            return False
        
        # Check OpenAlex rate limits if using OpenAlex as paper source
        if Settings.PAPER_SOURCE == "openalex":
            if not cls.check_openalex_rate_limits():
                return False
        
        print("[Pre-check] All checks passed.\n")
        return True
    
    @classmethod
    def run_all_checks_or_exit(cls):
        """Run all checks and exit if any fail."""
        if not cls.run_all_checks():
            sys.exit(1)
