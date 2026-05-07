"""Main orchestrator workflow for literature review generation."""

from workflows.summarization import summarization_workflow
from workflows.filter_papers import filter_papers_workflow
from workflows.literature_review_creation import literature_review_creation_flow
from workflows.revising_draft import revising_draft_workflow


async def literature_review_generator_workflow(topic: str, paper_titles: list[str]):
    """
    Main orchestrator workflow that chains all sub-workflows together.
    
    Workflow:
    1. Summarize papers by title
    2. Filter papers for relevance to topic
    3. Create initial literature review draft
    4. Allow user to revise draft iteratively
    
    Args:
        topic: The topic for the literature review
        paper_titles: List of paper titles to search for
        
    Returns:
        Final revised literature review as string
    """
    # Step 1: Summarize papers
    summary_of_papers = await summarization_workflow(paper_titles)
    assert summary_of_papers is not None, "Summarization workflow returned None"
    
    # Step 2: Filter papers by relevance
    filtered_summary_of_papers = await filter_papers_workflow(topic, summary_of_papers)
    assert filtered_summary_of_papers is not None, "Filter workflow returned None"
    
    # Step 3: Create initial literature review
    literature_review_draft = await literature_review_creation_flow(topic, filtered_summary_of_papers)
    assert literature_review_draft is not None, "Literature review creation returned None"
    
    # Step 4: Allow user to revise draft
    final_draft = await revising_draft_workflow(literature_review_draft)
    print("FINAL DRAFT: ")
    print(final_draft)
    return final_draft
