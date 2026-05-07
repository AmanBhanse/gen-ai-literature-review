"""Literature review creation workflow with writer and editor agents."""

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from config import LITERATURE_REVIEW_WORD_COUNT
from utils import extract_draft_from_message
from agents.literature_review_writer import create_literature_review_writer_agent
from agents.literature_review_editor import create_literature_review_editor_agent


def _get_writer_system_message() -> str:
    """Generate system message for literature review writer agent."""
    return f"""
You are a writer who helps write literature reviews for a given topic.
Write a literature review in exactly {LITERATURE_REVIEW_WORD_COUNT} words.
You can ask for feedback from the editor agent.
If the editor approves, end with 'TERMINATE' and provide your final draft as JSON.

Return responses as JSON:
{{
  "content": "your literature review text here",
  "word_count": [actual count],
  "status": "draft|final",
  "ready_for_review": true|false
}}
"""


def _get_editor_system_message() -> str:
    """Generate system message for literature review editor agent."""
    return f"""
You are an editor and knowledgeable researcher. 
Your role is to:
1. Review the writer's draft
2. Provide constructive feedback if needed
3. Ensure the review is {LITERATURE_REVIEW_WORD_COUNT} words and meets quality standards
4. Approve when complete or request changes

Return responses as JSON:
{{
  "content": "feedback or approval message",
  "approved": true|false,
  "word_count_ok": true|false,
  "status": "needs_revision|approved"
}}
"""


def _get_task_str(literature_review_topic: str, summary_context: str) -> str:
    """Generate task prompt for literature review creation."""
    return f"""
Write a literature review on "{literature_review_topic}" in exactly {LITERATURE_REVIEW_WORD_COUNT} words.

Use these papers as references:
{summary_context}

Return your response as JSON with "content" containing the review text.
"""


async def literature_review_creation_flow(literature_review_topic: str, summary_context: str):
    """
    Create a literature review with writer and editor agents collaborating.
    
    Args:
        literature_review_topic: The topic for the literature review
        summary_context: JSON with summarized papers to use as references
        
    Returns:
        JSON string with final literature review draft
    """
    writer_agent = create_literature_review_writer_agent(_get_writer_system_message())
    editor_agent = create_literature_review_editor_agent(_get_editor_system_message())
    
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=8)
    termination = text_mention_termination | max_messages_termination
    literature_review_creation_team = RoundRobinGroupChat([writer_agent, editor_agent], termination_condition=termination)
    await literature_review_creation_team.reset()
    task_result = await Console(literature_review_creation_team.run_stream(task=_get_task_str(literature_review_topic, summary_context)))
    messages = task_result.messages if hasattr(task_result, "messages") else []
    last_message = messages[-1] if messages else None
    final_draft = extract_draft_from_message(last_message)
    if final_draft is None and last_message:
        final_draft = last_message.content
    return final_draft
