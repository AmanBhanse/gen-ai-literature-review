# utils/extraction.py
"""Extract content from LLM responses."""

import re
import json
import logging

logger = logging.getLogger(__name__)


class ContentExtractor:
    """Extract content from LLM responses using multiple strategies."""
    
    @staticmethod
    def from_json(last_message):
        """
        Extract content from JSON-formatted LLM response.
        Expected format: {"content": "actual text", "status": "complete"}
        
        Returns:
            tuple: (content, success, method)
        """
        if not last_message:
            return None, False, "no_message"
        
        message_content = last_message.content.strip()
        
        try:
            # Find JSON object in the message (handles case where LLM adds extra text)
            json_match = re.search(r'\{.*\}', message_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                if "content" in parsed and parsed["content"]:
                    logger.info(f"✓ Successfully extracted via JSON method. Status: {parsed.get('status', 'unknown')}")
                    return parsed["content"].strip(), True, "json"
        except (json.JSONDecodeError, AttributeError) as e:
            logger.debug(f"JSON extraction failed: {e}")
        
        return None, False, "json_failed"
    
    @staticmethod
    def from_separators(last_message):
        """
        Extract content from separator-delimited LLM response.
        Fallback method for backward compatibility.
        
        Returns:
            tuple: (content, success, method)
        """
        if not last_message:
            return None, False, "no_message"
        
        try:
            message_content = last_message.content
            pattern = rf"OUTPUT : STARTS(.*?)OUTPUT : ENDS"
            match = re.search(pattern, message_content, re.DOTALL)
            
            if match:
                content = match.group(1).strip()
                if content:
                    logger.info("✓ Successfully extracted via SEPARATOR method (fallback)")
                    return content, True, "separator"
        except Exception as e:
            logger.debug(f"Separator extraction failed: {e}")
        
        return None, False, "separator_failed"
    
    @classmethod
    def extract(cls, last_message):
        """
        Extract content from LLM response using hybrid approach.
        Tries JSON first (preferred), falls back to separator format.
        
        Args:
            last_message: LLM message object with content attribute
            
        Returns:
            str: Extracted content or None if all methods fail
        """
        if not last_message:
            logger.warning("No message received from LLM")
            return None
        
        # Try JSON method first (preferred)
        content, success, method = cls.from_json(last_message)
        if success:
            return content
        
        # Fall back to separator method
        content, success, method = cls.from_separators(last_message)
        if success:
            return content
        
        # Both methods failed - log warning and attempt to get any substantial content
        logger.warning(f"✗ Failed to extract content using standard methods. Message preview: {last_message.content[:100]}...")
        
        # Last resort: check if message is very short and return it directly
        if last_message.content and len(last_message.content) > 10:
            logger.warning("Returning raw message content as last resort")
            return last_message.content
        
        return None


# Backward compatibility: functions that delegate to class
def extract_draft_from_message_json(last_message):
    """Extract content from JSON-formatted response (backward compatibility)."""
    return ContentExtractor.from_json(last_message)


def extract_draft_from_message_separator(last_message):
    """Extract content from separator-delimited response (backward compatibility)."""
    return ContentExtractor.from_separators(last_message)


def extract_draft_from_message(last_message):
    """Extract content from LLM response (backward compatibility)."""
    return ContentExtractor.extract(last_message)
