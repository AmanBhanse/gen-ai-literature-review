"""Summarization workflow for fetching and summarizing academic papers."""

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from config import SINGLE_PAPER_SUMMARY_WORD_COUNT
from utils import extract_draft_from_message, fetch_papers
from agents.summarization_agent import create_summarization_agent


def _paper_res_to_txt(papers: list) -> str:
    """Format papers list into readable text."""
    txt = ""
    for idx, paper in enumerate(papers):
        txt += f"{idx + 1} : Title : {paper['title']}\nDescription : {paper['summary']}\nLINK : {paper['link']}\n\n"
    return txt


def _prompt_for_summarization(papers: list) -> str:
    """Generate system prompt for summarization agent."""
    prompt = f"""
You are a knowledgeable research assistant who summarizes academic papers.
Below are the papers provided by the user. Summarize each paper in {SINGLE_PAPER_SUMMARY_WORD_COUNT} words using the URL link provided.

Return your response as JSON with the following structure:
{{
  "content": "paper_1:\\n- Title: [title]\\n- link: [link]\\n- Summary: [summary]\\n\\npaper_2:\\n- Title: [title]\\n- link: [link]\\n- Summary: [summary]\\n\\n...",
  "status": "complete",
  "count": [number of papers summarized]
}}

Papers to summarize:
{_paper_res_to_txt(papers)}

IMPORTANT: Your response must be valid JSON. Include all papers in the single "content" field."""
    return prompt


async def summarization_workflow(paper_titles: list):
    """
    Fetch papers by title and summarize each one.
    
    Args:
        paper_titles: List of paper titles to search for
        
    Returns:
        JSON string with summarized papers
    """
    print("- Fetching papers (using scholarly with caching)...")
    fetched_papers = []
    for paper in paper_titles:
        fetch_pap = fetch_papers(paper, 1)
        if len(fetch_pap) > 0:
            fetched_papers.append(fetch_pap[0])
            link = fetch_pap[0]["link"]
            source = fetch_pap[0].get("source", "unknown")
            print(f"{' '*3}+ {link} (source: {source})")
    
    summarization_agent = create_summarization_agent(_prompt_for_summarization(fetched_papers))
    
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=2)
    termination = text_mention_termination | max_messages_termination
    summarization_team = RoundRobinGroupChat([summarization_agent], termination_condition=termination)
    await summarization_team.reset()
    task_str = "Summarize the following papers:\n\n" + _paper_res_to_txt(fetched_papers)
    task_result = await Console(summarization_team.run_stream(task=task_str))
    messages = task_result.messages if hasattr(task_result, "messages") else []
    last_message = messages[-1] if messages else None
    output = extract_draft_from_message(last_message)
    return output
