# Agentic Research Assistant

This project is an advanced **Agentic RAG (Retrieval-Augmented Generation)** application. It transforms a standard linear RAG pipeline into an intelligent, stateful research assistant that plans, retrieves iteratively, reflects on evidence, and generates highly accurate answers.

**Key Features Include:**
- **Agentic Workflow:** LangGraph orchestrates planning, searching, and reflecting.
- **Targeted Retrieval:** Users can select exactly which uploaded documents the agent should focus on, preventing global search noise.
- **Transparent Reasoning:** The Streamlit UI actively displays the agent's internal workflow (Planning, Retrieving, Reflecting) and the LLM's raw `<think>` block logic.

Built with Python, Streamlit, Google Gemini, ChromaDB, and LangGraph.
## 1. Project Folder Structure

The project follows a modular structure, separating the core RAG components from the agentic workflow logic:

```text
rag_agentic/
│
├── app.py                  
├── requirements.txt
├── .env
│
├── chunking.py             
├── ingestion.py            
├── vectorstore.py           
├── llm.py                  
│
├── agent/                  
│   ├── graph.py            
│   ├── state.py             
│   └── nodes.py           
│
└── chroma_db/           
```

## 2. Execution Flow

The workflow moves from a linear process to a dynamic loop, managed by LangGraph:

1. **User Query & Context Selection**: The user submits a question via the Streamlit UI and optionally selects specific documents to query against.
2. **Planner Agent (`plan_node`)**:
   - Analyzes the query to understand intent (factual, summary, comparison).
   - Formulates a step-by-step reasoning plan.
   - Provides the initial search query.
3. **Retrieval Agent (`retrieve_node`)**:
   - Executes search queries against the `VectorStore`.
   - Fetches the `top-k` chunks.
   - Accumulates retrieved chunks into the agent's working memory, avoiding duplicates.
4. **Reflection Agent (`reflect_node`)**:
   - Compares the accumulated evidence against the initial query.
   - Decides if the evidence is sufficient to answer the question confidently.
   - **Condition**: 
     - If *Insufficient*: Generates an alternative search query and triggers another retrieval loop.
     - If *Sufficient*: Ends the graph execution.
5. **Response Generator (`app.py`)**:
   - Synthesizes the final answer using the collected evidence from the graph.
   - Streams the response to the user via the Streamlit UI.
6. **UI Update**: Streamlit displays the agent's thought process (planning, reflecting, and model reasoning) followed by the final streamed answer and retrieved sources.

