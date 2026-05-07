"""Filter papers workflow for relevance filtering."""

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from utils import extract_draft_from_message
from agents.filter_agent import create_filter_agent


def _system_msg_filter_agent(user_topic_on_literature_review: str) -> str:
    """Generate system message for filter agent."""
    system_msg = f"""
You are a knowledgeable researcher who filters papers based on relevance.
Your job is to identify and remove papers that are not relevant to the given topic.
Only keep papers that directly relate to: {user_topic_on_literature_review}
"""
    return system_msg


def _get_task_str(user_topic_on_literature_review: str, output_from_summarization_agent: str) -> str:
    """Generate task prompt for filter agent."""
    prompt = f"""
Given below are papers with summaries, titles, and online links:
{output_from_summarization_agent}

TASK: Filter these papers based on relevance to the topic: **{user_topic_on_literature_review}**

Remove papers that are not relevant. Keep only papers that directly relate to the topic.

Return your response as JSON:
{{
  "content": "paper_1:\\n- Title: [title]\\n- link: [link]\\n- Summary: [summary]\\n\\npaper_2:\\n- Title: [title]\\n- link: [link]\\n- Summary: [summary]\\n\\n...",
  "status": "complete",
  "papers_kept": [number],
  "papers_removed": [number]
}}

IMPORTANT: Your response must be valid JSON. Include only relevant papers in the "content" field.
"""
    return prompt


async def filter_papers_workflow(user_topic_on_literature_review: str, output_from_summarization_agent: str):
    """
    Filter papers based on relevance to a given topic.
    
    Args:
        user_topic_on_literature_review: The topic for filtering relevance
        output_from_summarization_agent: JSON output from summarization workflow
        
    Returns:
        JSON string with filtered papers
    """
    filter_agent = create_filter_agent(_system_msg_filter_agent(user_topic_on_literature_review))
    
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=2)
    termination = text_mention_termination | max_messages_termination
    filter_team = RoundRobinGroupChat([filter_agent], termination_condition=termination)
    await filter_team.reset()
    task_result = await Console(filter_team.run_stream(task=_get_task_str(user_topic_on_literature_review, output_from_summarization_agent)))
    messages = task_result.messages if hasattr(task_result, "messages") else []
    last_message = messages[-1] if messages else None
    output = extract_draft_from_message(last_message)
    return output
