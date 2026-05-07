"""Filter papers agent definition."""

from autogen_agentchat.agents import AssistantAgent
from llm_client import get_llama3_client


def create_filter_agent(system_message: str) -> AssistantAgent:
    """Create a filter agent for removing irrelevant papers."""
    return AssistantAgent(
        name="filter_agent",
        system_message=system_message,
        model_client=get_llama3_client(),
    )
