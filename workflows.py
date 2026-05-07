# workflows.py
import asyncio
from agents import get_llama3_client
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from config import LITERATURE_REVIEW_WORD_COUNT, SINGLE_PAPER_SUMMARY_WORD_COUNT
from utils import extract_draft_from_message, fetch_google_scholar_papers

# --- Summarization Workflow ---
async def summarization_workflow(paper_titles: list):
    def paper_res_to_txt(papers: list) -> str:
        txt = ""
        for idx, paper in enumerate(papers):
            txt += f"{idx + 1} : Title : {paper['title']}\nDescription : {paper['summary']}\nLINK : {paper['link']}\n\n"
        return txt
    
    def prompt_for_summarization(papers: list) -> str:
        prompt = f"""
You are a knowledgeable research assistant who summarizes academic papers.
Below are the papers provided by the user. Summarize each paper in {SINGLE_PAPER_SUMMARY_WORD_COUNT} words using the URL link provided.

Return your response as JSON with the following structure:
{{
  "content": "paper_1:\\n- Title: [title]\\n- link: [link]\\n- Summary: [summary]\\n\\npaper_2:\\n- Title: [title]\\n- link: [link]\\n- Summary: [summary]\\n\\n...",
  "status": "complete",
  "count": [number of papers summarized]
}}

Papers to summarize:
{paper_res_to_txt(papers)}

IMPORTANT: Your response must be valid JSON. Include all papers in the single "content" field."""
        return prompt
    
    print("- Fetching papers from Google Scholar...")
    fetched_papers = []
    for paper in paper_titles:
        fetch_pap = fetch_google_scholar_papers(paper, 1)
        if len(fetch_pap) > 0:
            fetched_papers.append(fetch_pap[0])
            link = fetch_pap[0]["link"]
            print(f"{' '*3}+ {link}")
    
    summarization_agent = AssistantAgent(
        name="summarization_agent",
        system_message=prompt_for_summarization(fetched_papers),
        model_client=get_llama3_client(),
    )
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=2)
    termination = text_mention_termination | max_messages_termination
    summarization_team = RoundRobinGroupChat([summarization_agent], termination_condition=termination)
    await summarization_team.reset()
    task_str = "Summarize the following papers:\n\n" + paper_res_to_txt(fetched_papers)
    task_result = await Console(summarization_team.run_stream(task=task_str))
    messages = task_result.messages if hasattr(task_result, "messages") else []
    last_message = messages[-1] if messages else None
    output = extract_draft_from_message(last_message)
    return output

# --- Filter Papers Workflow ---
async def filter_papers_workflow(user_topic_on_literature_review, output_from_summarization_agent: str):
    def system_msg_filter_agent() -> str:
        system_msg = f"""
You are a knowledgeable researcher who filters papers based on relevance.
Your job is to identify and remove papers that are not relevant to the given topic.
Only keep papers that directly relate to: {user_topic_on_literature_review}
"""
        return system_msg
    
    def get_task_str() -> str:
        prompt = f"""
Given below are papers with summaries, titles, and online links:
{output_from_summarization_agent}

TASK: Filter these papers based on relevance to the topic: **{user_topic_on_literature_review}**

Remove papers that are not relevant. Keep only papers that directly relate to the topic.

Return your response as JSON:
{{
  "content": "paper_1:\\n- Title: [title]\\n- link: [link]\\n- Summary: [summary]\\n\\npaper_2:\\n- Title: [title]\\n- link: [link]\\n- Summary: [summary]\\n\\n...",
  "status": "complete",
  "papers_kept": [number],
  "papers_removed": [number]
}}

IMPORTANT: Your response must be valid JSON. Include only relevant papers in the "content" field.
"""
        return prompt
    
    filter_agent = AssistantAgent(
        name="filter_agent",
        system_message=system_msg_filter_agent(),
        model_client=get_llama3_client(),
    )
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=2)
    termination = text_mention_termination | max_messages_termination
    filter_team = RoundRobinGroupChat([filter_agent], termination_condition=termination)
    await filter_team.reset()
    task_result = await Console(filter_team.run_stream(task=get_task_str()))
    messages = task_result.messages if hasattr(task_result, "messages") else []
    last_message = messages[-1] if messages else None
    output = extract_draft_from_message(last_message)
    return output

# --- Literature Review Creation Workflow ---
async def literature_review_creation_flow(literature_review_topic: str, summary_context: str):
    writer_agent = AssistantAgent(
        name="literature_review_writer",
        description="""Agent for writing literature review given by user.""",
        system_message=f"""
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
""",
        model_client=get_llama3_client(),
    )
    
    editor_agent = AssistantAgent(
        name="literature_review_editor",
        description="""Agent for editing and approving literature reviews.""",
        system_message=f"""
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
""",
        model_client=get_llama3_client(),
    )
    
    def get_task_str():
        return f"""
Write a literature review on "{literature_review_topic}" in exactly {LITERATURE_REVIEW_WORD_COUNT} words.

Use these papers as references:
{summary_context}

Return your response as JSON with "content" containing the review text.
"""
    
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=8)
    termination = text_mention_termination | max_messages_termination
    literature_review_creation_team = RoundRobinGroupChat([writer_agent, editor_agent], termination_condition=termination)
    await literature_review_creation_team.reset()
    task_result = await Console(literature_review_creation_team.run_stream(task=get_task_str()))
    messages = task_result.messages if hasattr(task_result, "messages") else []
    last_message = messages[-1] if messages else None
    final_draft = extract_draft_from_message(last_message)
    if final_draft is None and last_message:
        final_draft = last_message.content
    return final_draft

# --- Revising Draft Workflow ---
async def revising_draft_workflow(draft: str):
    writer_agent = AssistantAgent(
        name="literature_review_writer",
        description="""Agent for revising literature review drafts based on feedback.""",
        system_message=f"""
You are a writer who revises literature review drafts based on feedback.
Update the draft to incorporate user feedback while maintaining {LITERATURE_REVIEW_WORD_COUNT} words.

Return responses as JSON:
{{
  "content": "revised literature review text",
  "word_count": [actual count],
  "changes_made": "brief description of changes",
  "status": "revised"
}}
""",
        model_client=get_llama3_client(),
    )
    
    editor_agent = AssistantAgent(
        name="literature_review_editor",
        description="""Agent for reviewing and approving revised literature.""",
        system_message=f"""
You are an editor and knowledgeable researcher reviewing revisions.
Check if the user's requested changes were properly implemented.
Ensure the review remains {LITERATURE_REVIEW_WORD_COUNT} words and maintains quality.

Return responses as JSON:
{{
  "content": "feedback on revision",
  "approved": true|false,
  "word_count_ok": true|false
}}
""",
        model_client=get_llama3_client(),
    )
    
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

# --- Main Literature Review Generator Workflow ---
async def literature_review_generator_workflow(topic, paper_titles: list[str]):
    summary_of_papers = await summarization_workflow(paper_titles)
    assert summary_of_papers is not None, "Summarization workflow returned None"
    
    filtered_summary_of_papers = await filter_papers_workflow(topic, summary_of_papers)
    assert filtered_summary_of_papers is not None, "Filter workflow returned None"
    
    literature_review_draft = await literature_review_creation_flow(topic, filtered_summary_of_papers)
    assert literature_review_draft is not None, "Literature review creation returned None"
    
    final_draft = await revising_draft_workflow(literature_review_draft)
    print("FINAL DRAFT: ")
    print(final_draft)
    return final_draft
