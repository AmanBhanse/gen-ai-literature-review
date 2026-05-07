# llm_client.py
# LLM Client Configuration
# Provides a configured Groq API client for Llama 3.3-70b model

from autogen_ext.models.openai import OpenAIChatCompletionClient
from config_module import Settings


def get_llama3_client():
    """
    Get a configured Groq Llama 3.3-70b API client.
    
    Uses OpenAI-compatible API to connect to Groq's Llama model.
    Configured for JSON output, function calling, and chat completion.
    
    Returns:
        OpenAIChatCompletionClient: Configured client ready for use with AutoGen agents
    """
    model_client = OpenAIChatCompletionClient(
        model="llama-3.3-70b-versatile",
        base_url="https://api.groq.com/openai/v1",
        api_key=Settings.GROQ_API_KEY,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": "llama3",
        },
    )
    return model_client
