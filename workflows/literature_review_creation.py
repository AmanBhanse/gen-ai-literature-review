"""Literature review creation workflow with writer and editor agents."""

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from workflows.prompts import PromptGenerator
from utils import extract_draft_from_message
from agents.literature_review_writer import create_literature_review_writer_agent
from agents.literature_review_editor import create_literature_review_editor_agent


async def literature_review_creation_flow(literature_review_topic: str, summary_context: str):
    """
    Create a literature review with writer and editor agents collaborating.
    
    Args:
        literature_review_topic: The topic for the literature review
        summary_context: JSON with summarized papers to use as references
        
    Returns:
        JSON string with final literature review draft
    """
    prompts = PromptGenerator()
    
    writer_agent = create_literature_review_writer_agent(prompts.literature_review_writer_system_message())
    editor_agent = create_literature_review_editor_agent(prompts.literature_review_editor_system_message())
    
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=8)
    termination = text_mention_termination | max_messages_termination
    literature_review_creation_team = RoundRobinGroupChat([writer_agent, editor_agent], termination_condition=termination)
    await literature_review_creation_team.reset()
    task_result = await Console(literature_review_creation_team.run_stream(task=prompts.literature_review_creation_task(literature_review_topic, summary_context)))
    messages = task_result.messages if hasattr(task_result, "messages") else []
    last_message = messages[-1] if messages else None
    final_draft = extract_draft_from_message(last_message)
    if final_draft is None and last_message:
        final_draft = last_message.content
    return final_draft
