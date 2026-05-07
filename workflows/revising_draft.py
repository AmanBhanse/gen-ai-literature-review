"""Revising draft workflow for iterative literature review refinement."""

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from config import LITERATURE_REVIEW_WORD_COUNT
from utils import extract_draft_from_message
from agents.literature_review_writer import create_literature_review_writer_agent
from agents.literature_review_editor import create_literature_review_editor_agent


def _get_revision_writer_system_message() -> str:
    """Generate system message for revision writer agent."""
    return f"""
You are a writer who revises literature review drafts based on feedback.
Update the draft to incorporate user feedback while maintaining {LITERATURE_REVIEW_WORD_COUNT} words.

Return responses as JSON:
{{
  "content": "revised literature review text",
  "word_count": [actual count],
  "changes_made": "brief description of changes",
  "status": "revised"
}}
"""


def _get_revision_editor_system_message() -> str:
    """Generate system message for revision editor agent."""
    return f"""
You are an editor and knowledgeable researcher reviewing revisions.
Check if the user's requested changes were properly implemented.
Ensure the review remains {LITERATURE_REVIEW_WORD_COUNT} words and maintains quality.

Return responses as JSON:
{{
  "content": "feedback on revision",
  "approved": true|false,
  "word_count_ok": true|false
}}
"""


async def revising_draft_workflow(draft: str):
    """
    Iteratively revise a literature review draft based on user feedback.
    
    Args:
        draft: The initial draft to revise
        
    Returns:
        Revised final draft after user feedback loop
    """
    writer_agent = create_literature_review_writer_agent(_get_revision_writer_system_message())
    editor_agent = create_literature_review_editor_agent(_get_revision_editor_system_message())
    
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=8)
    termination = text_mention_termination | max_messages_termination
    revising_team = RoundRobinGroupChat([editor_agent, writer_agent], termination_condition=termination)
    await revising_team.reset()
    revised_final_draft = draft
    
    while True:
        user_task = input("Enter your feedback (type 'exit' to leave): ")
        if user_task.lower().strip() == "exit":
            print("- No additional changes requested by user. Exiting...")
            break
        
        task_str = f"""
User feedback: {user_task}

Current draft:
{revised_final_draft}

Please revise the draft based on the user feedback.
"""
        task_result = await Console(revising_team.run_stream(task=task_str))
        messages = task_result.messages if hasattr(task_result, "messages") else []
        last_message = messages[-1] if messages else None
        
        if last_message:
            revised_final_draft = extract_draft_from_message(last_message)
            if revised_final_draft is None:
                revised_final_draft = last_message.content
    
    return revised_final_draft
