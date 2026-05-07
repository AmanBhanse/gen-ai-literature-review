"""Filter papers workflow for relevance filtering."""

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from workflows.prompts import PromptGenerator
from utils import extract_draft_from_message
from agents.filter_agent import create_filter_agent


async def filter_papers_workflow(user_topic_on_literature_review: str, output_from_summarization_agent: str):
    """
    Filter papers based on relevance to a given topic.
    
    Args:
        user_topic_on_literature_review: The topic for filtering relevance
        output_from_summarization_agent: JSON output from summarization workflow
        
    Returns:
        JSON string with filtered papers
    """
    prompts = PromptGenerator()
    
    filter_agent = create_filter_agent(prompts.filter_system_message(user_topic_on_literature_review))
    
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=2)
    termination = text_mention_termination | max_messages_termination
    filter_team = RoundRobinGroupChat([filter_agent], termination_condition=termination)
    await filter_team.reset()
    task_result = await Console(filter_team.run_stream(task=prompts.filter_task(user_topic_on_literature_review, output_from_summarization_agent)))
    messages = task_result.messages if hasattr(task_result, "messages") else []
    last_message = messages[-1] if messages else None
    output = extract_draft_from_message(last_message)
    return output
