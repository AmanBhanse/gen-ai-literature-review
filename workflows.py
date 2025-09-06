# workflows.py
import asyncio
from agents import get_llama3_client
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from config import OUTPUT_SEPERATOR_START, OUTPUT_SEPERATOR_END, LITERATURE_REVIEW_WORD_COUNT, SINGLE_PAPER_SUMMARY_WORD_COUNT
from utils import extract_draft_from_message, fetch_google_scholar_papers

# --- Summerization Workflow ---
async def summerization_workflow(paper_titles: list):
    def paper_res_to_txt(papers: list) -> str:
        txt = ""
        for idx, paper in enumerate(papers):
            txt += f"{idx + 1} : Title : {paper['title']}\nDescription : {paper['summary']}\nLINK : {paper['link']}\n\n"
        return txt
    def prompt_for_summerization(papers: list) -> str:
        prompt = f"""
        You are Knowledgeable Research Asistant who can want to summary of each paper.
        Below are the papers given by the user. Summerize each paper in {SINGLE_PAPER_SUMMARY_WORD_COUNT} words using URL link to the paper provided.
        summerize them in following format:\n\n
        {OUTPUT_SEPERATOR_START}
        {{paper 1}}:
        - Title: {{title}}
        - link: {{link}}
        - Summary: {{summarization of whole paper based on your understanding}}
        {{paper 2}}:
        Title: {{title}}
        link: {{link}}
        Summary: {{summarization of whole paper based on your understanding}}
        so on...
        {OUTPUT_SEPERATOR_END}
        MAKE TO USE THE ABOVE TEMPLATE TO GIVE OUTPUT. i.e. seperator {OUTPUT_SEPERATOR_START} and {OUTPUT_SEPERATOR_END} must be present
        """
        return prompt
    print("- fetching papers from google scholar")
    fetched_papers = []
    for paper in paper_titles:
        fetch_pap = fetch_google_scholar_papers(paper, 1)
        if len(fetch_pap) > 0:
            fetched_papers.append(fetch_pap[0])
            link = fetch_pap[0]["link"]
            print(f"{' '*3}+ {link}")
    summerization_agent = AssistantAgent(
        name="summerization_agent",
        system_message=prompt_for_summerization(fetched_papers),
        model_client=get_llama3_client(),
    )
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=2)
    termination = text_mention_termination | max_messages_termination
    summerization_team = RoundRobinGroupChat([summerization_agent],termination_condition=termination)
    await summerization_team.reset()
    task_str = "Give summerization of following papers :\n\n" + paper_res_to_txt(fetched_papers)
    task_result = await Console(summerization_team.run_stream(task=task_str))
    messages = task_result.messages if hasattr(task_result, "messages") else []
    last_message = messages[-1] if messages else None
    output = extract_draft_from_message(last_message)
    return output

# --- Filter Papers Workflow ---
async def filter_papers_workflow(user_topic_on_literature_review, output_from_summerization_agent: str):
    def system_msg_filter_agent() -> str:
        system_msg = f"""
        You are a knowleagable Researcher who can filter the papers based on the summerization provided by the summerization agent.
        Your job is to Filter the papers which are not relevant to the user topic on literature review.
        If there are any papers which are not relevant to the user topic on literature review, remove them from the list.
        """
        return system_msg
    def get_task_str() -> str:
        prompt = f"""
        Given below are the list of some papers with inforamation like summary, title, online link:
        {output_from_summerization_agent}
        TASK :
        Take a deep look on papers by visiting the link provided.
        you need to remove the papers which are not relevant the user topic **{user_topic_on_literature_review}**
        OUTPUT should contain papers which are relevant in following template :
        {OUTPUT_SEPERATOR_START}
        {{paper 1}}:
        - Title: {{title}}
        - link: {{link}}
        - Summary: {{summarization of whole paper based on your understanding}}
        {{paper 2}}:
        Title: {{title}}
        link: {{link}}
        Summary: {{summarization of whole paper based on your understanding}}
        so on...
        {OUTPUT_SEPERATOR_END}
        MAKE TO USE THE ABOVE TEMPLATE TO GIVE OUTPUT. i.e. seperator {OUTPUT_SEPERATOR_START} and {OUTPUT_SEPERATOR_END} must be present
        """
        return prompt
    filter_agent = AssistantAgent(
        name="summerization_agent",
        system_message=system_msg_filter_agent(),
        model_client=get_llama3_client(),
    )
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=2)
    termination = text_mention_termination | max_messages_termination
    filter_team = RoundRobinGroupChat([filter_agent],termination_condition=termination)
    await filter_team.reset()
    task_result = await Console(filter_team.run_stream(task=get_task_str()))
    messages = task_result.messages if hasattr(task_result, "messages") else []
    last_message = messages[-1] if messages else None
    output = extract_draft_from_message(last_message)
    return output

# --- Literature Review Creation Workflow ---
async def literature_review_creation_flow(literature_review_topic: str, summary_conext: str):
    writer_agent = AssistantAgent(
        name="literature_review_writer",
        description="""
        Agent for writing literature review given by user.
        This agent should be the first to engage when given a new task.
        """,
        system_message=f"""
        You are a writter who helps write literature reviews for the user on the given topic.
        Make sure the literature review is writen in {LITERATURE_REVIEW_WORD_COUNT} words.
        you are allowed to ask feedback from the editor agent.
        If editor agent ask for changes, then make the changes and ask for approval again.
        If editor agent approves the task, end the conversation by saying **'TERMINATE'** and by giving final draft in following template:
        {OUTPUT_SEPERATOR_START}
        Write the final draft here.
        {OUTPUT_SEPERATOR_END}
        .
        DO NOT FORGOT MAKE TO USE THE ABOVE TEMPLATE TO GIVE FINAL OUTPUT. i.e. seperator {OUTPUT_SEPERATOR_START} and {OUTPUT_SEPERATOR_END} must be present
        """,
        model_client=get_llama3_client(),
    )
    editor_agent = AssistantAgent(
        name="literature_review_editor",
        description="""
        Agent for editing literature review given by user.
        This agent should be the second to engage when given a new task.
        """,
        system_message="""
        You are an Editor who is a knowleagable researcher too. Plan and guide the task given by the user. Provide critical feedbacks to the draft proceduced by Writer."
        Approve if the task is completed and the draft meets user's requirements.
        """,
        model_client=get_llama3_client(),
    )
    def get_task_str():
        return f"""
        write a literature review on {literature_review_topic} in {LITERATURE_REVIEW_WORD_COUNT} words.
        PLEASE PROVIDE THE FINAL OUTPUT IN FOLLOWING TEMPLATE:
        {OUTPUT_SEPERATOR_START}
        Write the final draft here.
        {OUTPUT_SEPERATOR_END}
        Please use the following papers in drafting the final literature review:
        {summary_conext}
        """
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=8)
    termination = text_mention_termination | max_messages_termination
    literature_review_creation_team = RoundRobinGroupChat([writer_agent,editor_agent],termination_condition=termination)
    await literature_review_creation_team.reset()
    task_result = await Console(literature_review_creation_team.run_stream(task=get_task_str()))
    messages = task_result.messages if hasattr(task_result, "messages") else []
    last_message = messages[-1] if messages else None
    final_draft = extract_draft_from_message(last_message)
    if final_draft == None:
        final_draft = last_message.content
    return final_draft

# --- Revising Draft Workflow ---
async def revising_draft_workflow(draft: str):
    writer_agent = AssistantAgent(
        name="literature_review_writer",
        description="""
        Agent for editing literature review draft given by user.
        """,
        system_message=f"""
        You are a writter who is going to make changes to the given literature review draft.
        Make sure the revised literature review is writen in {LITERATURE_REVIEW_WORD_COUNT} words.
        Editor will ask you to make changes to the draft.
        If editor agent ask for changes, then make the appropriate changes and ask for approval again.
        If editor agent approves the task, end the conversation by saying **'TERMINATE'** and by giving final draft in following template:
        {OUTPUT_SEPERATOR_START}
        Write the final draft here.
        {OUTPUT_SEPERATOR_END}
        .
        MAKE TO USE THE ABOVE TEMPLATE TO GIVE FINAL DRAFT. i.e. seperator {OUTPUT_SEPERATOR_START} and {OUTPUT_SEPERATOR_END} must be present
        """,
        model_client=get_llama3_client(),
    )
    editor_agent = AssistantAgent(
        name="literature_review_editor",
        description="""
        Agent for editing literature review given by user.
        This agent should be the first to engage when given a new task.
        """,
        system_message=f"""
        You are an Editor who is a knowleagable researcher too.
        User has given one literature review draft to you, and asking to make minor changes.
        Your job is to understand what changes user want in the literature review draft and instruct the writer agent to make appropriate changes.
        Make sure that final writer agent writes the review in {LITERATURE_REVIEW_WORD_COUNT} words.
        If Writer agent haven't made changes to draft correctly, provide appropriate feedback to writer agent.
        Else Approve if the task is completed and the draft meets user's requirements.
        """,
        model_client=get_llama3_client(),
    )
    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=8)
    termination = text_mention_termination | max_messages_termination
    revising_team = RoundRobinGroupChat([editor_agent, writer_agent],termination_condition=termination)
    await revising_team.reset()
    revised_final_draft = draft
    task_result = None
    while True:
        user_task = input("Enter your feedback (type 'exit' to leave): ")
        if user_task.lower().strip() == "exit":
            print("- No additional changes requested by user, EXISTING....")
            break
        task_str = f"""
        Please make the changes to the draft
        User Change request : {user_task}
        Draft to edit :
        {revised_final_draft}
        """
        task_result = await Console(revising_team.run_stream(task=task_str))
        messages = task_result.messages if hasattr(task_result, "messages") else []
        last_message = messages[-1] if messages else None
        revised_final_draft = extract_draft_from_message(last_message)
        if revised_final_draft == None:
            revised_final_draft = last_message.content
    return revised_final_draft

# --- Main Literature Review Generator Workflow ---
async def literature_review_generator_workflow(topic, paper_titles: list[str]):
    summary_of_papers = await summerization_workflow(paper_titles)
    assert summary_of_papers != None
    filtered_summary_of_papers = await filter_papers_workflow(topic, summary_of_papers)
    assert filtered_summary_of_papers != None
    literature_review_draft = await literature_review_creation_flow(topic, filtered_summary_of_papers)
    assert literature_review_draft != None
    final_draft = await revising_draft_workflow(literature_review_draft)
    print("FINAL DRAFT : ")
    print(final_draft)
    return final_draft
