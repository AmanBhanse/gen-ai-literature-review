"""
main.py
--------
Entry point for the Gen-AI Literature Review system.
This script demonstrates the full workflow using the modularized codebase.
"""

# 0. Install required packages (if running in a fresh environment)
# Recommended: Use requirements.txt for reproducibility
# Example: pip install -r requirements.txt

# 1. Imports Section
from config import (
    GROQ_API_KEY,
    OUTPUT_SEPERATOR_START,
    OUTPUT_SEPERATOR_END,
    LITERATURE_REVIEW_WORD_COUNT,
    SINGLE_PAPER_SUMMARY_WORD_COUNT
)
from utils import extract_draft_from_message, fetch_google_scholar_papers
from agents import get_llama3_client
from workflows import (
    summerization_workflow,
    filter_papers_workflow,
    literature_review_creation_flow,
    revising_draft_workflow,
    literature_review_generator_workflow
)
from metrics import calculate_rouge_score
import asyncio
import sys
import openai


# --- Pre-check Section ---
def precheck():
    """
    Checks for required configuration and API key validity.
    """
    print("\n[Pre-check] Verifying configuration and API access...")
    # Check GROQ API Key
    if not GROQ_API_KEY or not isinstance(GROQ_API_KEY, str) or not GROQ_API_KEY.startswith("gsk_"):
        print("[ERROR] GROQ_API_KEY is missing or invalid. Please update config.py with a valid key.")
        sys.exit(1)
    # Try a simple OpenAI API call to check key validity (using groq endpoint)
    try:
        client = openai.OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        # List models as a lightweight test
        _ = client.models.list()
        print("[OK] GROQ API key is valid and Groq endpoint is reachable.")
    except Exception as e:
        print(f"[ERROR] Failed to validate GROQ API key or reach Groq endpoint: {e}")
        sys.exit(1)
    print("[Pre-check] All checks passed.\n")


def main():
    precheck()
    """
    Main function to run the literature review workflow and compare with human-written review.
    """
    # 2. Define the literature review topic and paper titles
    literature_topic = "Generative AI"
    paper_titles = [
        "Advancements in Generative AI: A Comprehensive Review of GANs, GPT, Autoencoders, Diffusion Model, and Transformers Staphord Bengesi",
        "The Age of Generative AI and AI-Generated Everything Hongyang Du",
        "Generative AI: A Review on Models and Applications Kuldeep Singh Kaswan Jagjit Singh Dhatterwal",
        "At the Dawn of Generative AI Era: A Tutorial-cum-Survey on New Frontiers in 6G Wireless Intelligence Abdulkadir Celik",
        "The Internet of Things in the Era of Generative AI: Vision and Challenges Xin Wang",
        "Accelerating Innovation With Generative AI: AI-Augmented Digital Prototyping and Innovation Methods Volker Bilgram"
    ]

    # 3. Run the literature review generator workflow (async)
    print("\nRunning literature review generator workflow...")
    ai_gen_literature = asyncio.run(literature_review_generator_workflow(literature_topic, paper_titles))
    print("\nAI-Generated Literature Review:\n")
    print(ai_gen_literature)

    # 4. Human-written reference for metric comparison
    human_written = """
Generative AI, driven by Large Language Models (LLMs), has significantly advanced multi-agent systems (MAS), enabling more intelligent and autonomous operations across various domains.
 The integration of LLMs within MAS has led to enhanced reasoning, planning, and decision-making capabilities,
 positioning them as a promising pathway toward achieving artificial general intelligence. Research on LLM-based MAS highlights their applications in
 communication networks, problem-solving, world simulation, and real-time decision-making. One notable study introduces CommLLM, a multi-agent framework for
 optimizing 6G communications by employing LLMs to retrieve, plan, evaluate, and refine communication strategies through natural language processing. However,
 while LLMs offer substantial improvements in agent collaboration, they face challenges such as data scarcity in specialized domains, limited logical reasoning,
 and the need for better memory and evaluation mechanisms. To address these issues, AutoGen, an open-source framework, enables the construction of LLM-powered multi-agent
 conversational systems, enhancing inter-agent communication and adaptability. Another survey systematically examines LLM-driven MAS, identifying key components like profile,
 perception, self-action, mutual interaction, and evolution as foundational elements for their development. These systems are particularly effective in handling complex tasks
 that require collaboration, such as power grid management and traffic control, where single-agent models fall short. Despite these advancements, critical challenges remain,
 including optimizing task allocation, enhancing reasoning through iterative debates, and managing intricate memory structures to support dynamic interactions. The concept of
 self-adaptive MAS, powered by LLMs, presents a solution by incorporating autonomic computing principles where agents monitor and adjust their behaviors based on real-time concerns.
 This adaptive approach improves communication expressiveness and coordination, reducing the challenges of managing multiple interacting agents in dynamic environments. Additionally,
 LLM-based MAS have demonstrated potential in distributed systems like blockchain, showcasing their versatility in securing and optimizing decentralized networks. While the research
 emphasizes the strengths of these systems, it also underscores the limitations of LLMs in handling layered contexts and their reliance on predefined architectures for decision-making.
 Future work aims to refine these models by incorporating more robust evaluation frameworks, improving memory retention, and enhancing the interpretability of LLM-driven decisions.
 As these systems continue to evolve, their ability to simulate human-like intelligence, reason through complex problems, and interact autonomously will determine their effectiveness
 in real-world applications. The growing body of research on LLM-powered MAS reflects a paradigm shift in AI-driven automation, offering novel insights into the collaborative potential
 of intelligent agents while paving the way for future innovations in generative AI.
"""

    # 5. Metric comparison (ROUGE)
    print("\nComparing AI-generated review with human-written review using ROUGE metrics...")
    rouge_scores = calculate_rouge_score(ai_gen_literature, human_written)
    print("\nROUGE Scores:")
    print(rouge_scores)


if __name__ == "__main__":
    main()
