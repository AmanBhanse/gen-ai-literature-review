# utils.py
import re
from scholarly import scholarly
from config import OUTPUT_SEPERATOR_START, OUTPUT_SEPERATOR_END

def extract_draft_from_message(last_message):
    if last_message:
        last_message_content = last_message.content
        pattern = rf"OUTPUT : STARTS(.*?)OUTPUT : ENDS"
        match = re.search(pattern, last_message_content, re.DOTALL)
        return match.group(1).strip() if match else None
    return None

def fetch_google_scholar_papers(query, num_papers):
    """
    Fetches num_papers research papers from Google Scholar.
    Returns:
        list: A list of dictionaries containing paper details (title, summary, link)
    """
    papers = []
    search_results = scholarly.search_pubs(query)
    for i, paper in enumerate(search_results):
        if i >= num_papers:
            break
        papers.append({
            "title": paper["bib"]["title"],
            "summary": paper["bib"].get("abstract", "No summary available"),
            "link": paper.get("pub_url", "No link available")
        })
    return papers
