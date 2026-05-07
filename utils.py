# utils.py
import re
import json
import logging
from scholarly import scholarly
from config import DEBUG_MODE

# Configure logging based on DEBUG_MODE
log_level = logging.INFO if DEBUG_MODE else logging.WARNING
logging.basicConfig(level=log_level, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

def extract_draft_from_message_json(last_message):
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


def extract_draft_from_message_separator(last_message):
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


def extract_draft_from_message(last_message):
    """
    Extract content from LLM response using hybrid approach.
    Tries JSON first (preferred), falls back to separator format.
    
    Returns:
        str: Extracted content or None if all methods fail
    """
    if not last_message:
        logger.warning("No message received from LLM")
        return None
    
    # Try JSON method first (preferred)
    content, success, method = extract_draft_from_message_json(last_message)
    if success:
        return content
    
    # Fall back to separator method
    content, success, method = extract_draft_from_message_separator(last_message)
    if success:
        return content
    
    # Both methods failed - log warning and attempt to get any substantial content
    logger.warning(f"✗ Failed to extract content using standard methods. Message preview: {last_message.content[:100]}...")
    
    # Last resort: check if message is very short and return it directly
    if last_message.content and len(last_message.content) > 10:
        logger.warning("Returning raw message content as last resort")
        return last_message.content
    
    return None


def fetch_google_scholar_papers(query, num_papers):
    """
    Fetches num_papers research papers from Google Scholar.
    Returns:
        list: A list of dictionaries containing paper details (title, summary, link)
    """
    papers = []
    try:
        search_results = scholarly.search_pubs(query)
        for i, paper in enumerate(search_results):
            if i >= num_papers:
                break
            papers.append({
                "title": paper["bib"]["title"],
                "summary": paper["bib"].get("abstract", "No summary available"),
                "link": paper.get("pub_url", "No link available")
            })
        logger.info(f"Successfully fetched {len(papers)} papers for query: '{query}'")
    except Exception as e:
        logger.error(f"Error fetching papers from Google Scholar: {e}")
    
    return papers
