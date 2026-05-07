"""Revising draft workflow for iterative literature review refinement."""

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from workflows.prompts import PromptGenerator
from utils import extract_draft_from_message
from agents.literature_review_writer import create_literature_review_writer_agent
from agents.literature_review_editor import create_literature_review_editor_agent


async def revising_draft_workflow(draft: str):
    """
    Iteratively revise a literature review draft based on user feedback.
    
    Args:
        draft: The initial draft to revise
        
    Returns:
        Revised final draft after user feedback loop
    """
    prompts = PromptGenerator()
    
    writer_agent = create_literature_review_writer_agent(prompts.revision_writer_system_message())
    editor_agent = create_literature_review_editor_agent(prompts.revision_editor_system_message())
    
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
