"""Literature review writer agent definition."""

from autogen_agentchat.agents import AssistantAgent
from llm_client import get_llama3_client


def create_literature_review_writer_agent(system_message: str) -> AssistantAgent:
    """Create a writer agent for literature reviews."""
    return AssistantAgent(
        name="literature_review_writer",
        description="Agent for writing or revising literature reviews based on feedback.",
        system_message=system_message,
        model_client=get_llama3_client(),
    )
