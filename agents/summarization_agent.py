"""Summarization agent definition."""

from autogen_agentchat.agents import AssistantAgent
from llm_client import get_llama3_client


def create_summarization_agent(system_message: str) -> AssistantAgent:
    """Create a summarization agent."""
    return AssistantAgent(
        name="summarization_agent",
        system_message=system_message,
        model_client=get_llama3_client(),
    )
