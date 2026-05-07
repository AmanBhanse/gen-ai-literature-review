"""Literature review editor agent definition."""

from autogen_agentchat.agents import AssistantAgent
from llm_client import get_llama3_client


def create_literature_review_editor_agent(system_message: str) -> AssistantAgent:
    """Create an editor agent for reviewing literature reviews."""
    return AssistantAgent(
        name="literature_review_editor",
        description="Agent for editing and approving literature reviews.",
        system_message=system_message,
        model_client=get_llama3_client(),
    )
