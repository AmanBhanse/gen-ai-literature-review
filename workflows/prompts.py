# workflows/prompts.py
"""Centralized system messages and task prompts for all workflows."""

from config import LITERATURE_REVIEW_WORD_COUNT, SINGLE_PAPER_SUMMARY_WORD_COUNT


class PromptGenerator:
    """Generate system messages and task prompts for all workflows."""
    
    # ============================================================================
    # Summarization Prompts
    # ============================================================================
    
    @staticmethod
    def summarization_system_message() -> str:
        """Generate system message for summarization agent."""
        return """
You are a summarization agent. Your task is to summarize academic papers.
Keep summaries concise and informative.
Focus on key findings and contributions.
"""
    
    @staticmethod
    def summarization_task(papers: str) -> str:
        """Generate task prompt for summarization."""
        return f"""
Summarize the following papers in {SINGLE_PAPER_SUMMARY_WORD_COUNT} words each:

{papers}

Provide the summaries in JSON format with "papers" key containing an array of summaries.
"""
    
    # ============================================================================
    # Filter Agent Prompts
    # ============================================================================
    
    @staticmethod
    def filter_system_message(topic: str) -> str:
        """Generate system message for filter agent."""
        return f"""
You are an expert researcher filtering papers for relevance to: "{topic}"
Your role is to identify papers that are most relevant to this topic.
Be strict but fair in your filtering.
"""
    
    @staticmethod
    def filter_task(topic: str, summary_context: str) -> str:
        """Generate task prompt for filtering workflow."""
        return f"""
Filter these paper summaries to keep only those relevant to: "{topic}"

Here are the summaries:
{summary_context}

Return a JSON object with "filtered_papers" containing only the relevant ones.
"""
    
    # ============================================================================
    # Literature Review Writer Prompts
    # ============================================================================
    
    @staticmethod
    def literature_review_writer_system_message() -> str:
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
    
    @staticmethod
    def literature_review_editor_system_message() -> str:
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
    
    @staticmethod
    def literature_review_creation_task(topic: str, summary_context: str) -> str:
        """Generate task prompt for literature review creation."""
        return f"""
Write a literature review on "{topic}" in exactly {LITERATURE_REVIEW_WORD_COUNT} words.

Use these papers as references:
{summary_context}

Return your response as JSON with "content" containing the review text.
"""
    
    # ============================================================================
    # Revision Prompts
    # ============================================================================
    
    @staticmethod
    def revision_writer_system_message() -> str:
        """Generate system message for revision writer agent."""
        return f"""
You are a writer helping revise a literature review.
Improve the existing draft based on feedback.
Write in exactly {LITERATURE_REVIEW_WORD_COUNT} words.
Ask the editor for approval when ready.

Return responses as JSON:
{{
  "content": "revised literature review text here",
  "word_count": [actual count],
  "status": "draft|final"
}}
"""
    
    @staticmethod
    def revision_editor_system_message() -> str:
        """Generate system message for revision editor agent."""
        return f"""
You are an editor reviewing a revised literature review.
Check that it's {LITERATURE_REVIEW_WORD_COUNT} words and meets quality standards.
Provide feedback or approve.

Return responses as JSON:
{{
  "content": "feedback or approval message",
  "approved": true|false,
  "suggestions": "any improvement suggestions"
}}
"""
    
    @staticmethod
    def revision_task(draft: str) -> str:
        """Generate task prompt for revision workflow."""
        return f"""
Here is the current literature review draft:

{draft}

Please revise and improve it. Maintain approximately {LITERATURE_REVIEW_WORD_COUNT} words.

Return your revision as JSON with "content" containing the revised text.
"""
