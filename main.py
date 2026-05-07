"""
main.py
--------
Entry point for the Gen-AI Literature Review system.

This script orchestrates the full workflow:
1. Validates configuration and API keys
2. Fetches and summarizes papers
3. Filters papers for topic relevance
4. Generates a literature review using multi-agent collaboration
5. Allows iterative refinement
6. Compares with human-written review using ROUGE metrics
"""

import asyncio
from cli import Validator
from workflows import literature_review_generator_workflow


def main():
    """
    Main entry point for the literature review generation workflow.
    """
    # Validate configuration
    Validator.run_all_checks_or_exit()
    
    # Define the literature review topic and paper titles
    literature_topic = "Generative AI"
    paper_titles = [
        "Advancements in Generative AI: A Comprehensive Review of GANs, GPT, Autoencoders, Diffusion Model, and Transformers Staphord Bengesi",
        "The Age of Generative AI and AI-Generated Everything Hongyang Du",
        "Generative AI: A Review on Models and Applications Kuldeep Singh Kaswan Jagjit Singh Dhatterwal",
        "At the Dawn of Generative AI Era: A Tutorial-cum-Survey on New Frontiers in 6G Wireless Intelligence Abdulkadir Celik",
    ]

    # Run the literature review generator workflow
    print("\nRunning literature review generator workflow...")
    ai_gen_literature = asyncio.run(literature_review_generator_workflow(literature_topic, paper_titles))
    print("\nAI-Generated Literature Review:\n")
    print(ai_gen_literature)

    # Save final draft to output.txt
    print("\nSaving final draft to output.txt...")
    with open("output.txt", "w") as f:
        f.write(ai_gen_literature)
    print("✓ Final draft saved to output.txt")


if __name__ == "__main__":
    main()
