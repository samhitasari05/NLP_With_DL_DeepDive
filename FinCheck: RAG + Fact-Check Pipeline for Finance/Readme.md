**FinCheck: Financial Claim Verification Pipeline**

FinCheck is a retrieval-augmented, multi-agent fact-checking system designed to verify the credibility of financial claims using a layered architecture of retrieval, language modeling, and confidence-based decision logic. This project was developed to address the growing challenge of financial misinformation by combining structured knowledge retrieval with advanced generative and verification techniques.

Project Objective

The goal of this project is to automatically assess the truthfulness of financial statements. This is achieved through a sequence of steps:

Retrieving relevant context from a curated financial knowledge base.
Using a language model (LLM) to generate an informed response.
Passing the LLM output through a fact-checking agent that validates the claim using external data or logic.
Generating a final verdict using a voting mechanism that weighs both sources and their confidence levels.
This system enables real-time or batch processing of financial statements and aims to support news readers, finance professionals, and AI content moderation systems.

Key Components

1. Retrieval-Augmented Generation (RAG)
FinCheck uses vector search (via FAISS) to retrieve relevant documents or context paragraphs for a given input claim. These are then used to augment the prompt sent to the language model for grounded generation.

2. Language Model Inference
The retrieved content is passed to an LLM (e.g., OpenAI GPT), which evaluates the claim in context and provides a preliminary assessment along with reasoning.

3. Fact-Checking Agent
An independent second agent verifies the LLM’s judgment using additional lookups, pattern validation, or rules. This agent may use curated datasets, benchmark values (e.g., CPI, GDP), or alternate sources to cross-validate the model’s response.

4. Confidence Voting Logic
A weighted voting system assigns confidence scores to both LLM and fact-checking outputs. The final label—“True,” “False,” or “Uncertain”—is derived from thresholding and aggregate reliability checks.

Technologies Used

Python: Primary language used for all components
LangChain: For orchestration of the RAG pipeline
OpenAI GPT API: For generation and reasoning
FAISS: For vector-based document retrieval
Post-processing Logic: For confidence scoring and voting
Folder Structure

Sample Use Case

Input:
“The U.S. inflation rate dropped to 1.2% in Q2 2023.”

Pipeline Output:

Label: False
Reason: Retrieved benchmark data from U.S. BLS indicates an actual rate of 4.1% during the specified period.
Confidence: High
How to Run This Project

Clone the repository to your local machine.
Ensure Python 3.8+ is installed and set up a virtual environment.
Install dependencies via pip install -r requirements.txt.
Set your OpenAI API key in a .env file or directly in your scripts.
Run fincheck_pipeline.py with sample financial claims.
Review the generated outputs, which will include reasoning and confidence scores.
Future Directions

Integration with external financial APIs (e.g., World Bank, IMF) for dynamic grounding.
Improving fact-checker modularity to support domain-specific rules (e.g., tax law, ESG metrics).
Enhancing interpretability by attaching document citations and paragraph-level justifications.
License and Use

This project is open for educational and non-commercial use. Please cite the GitHub repository if you use or modify the FinCheck system for research or analysis.


