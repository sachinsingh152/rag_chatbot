import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from .state import AgentState

class PlannerOutput(BaseModel):
    plan: str = Field(description="Step by step reasoning plan to answer the user's query.")
    search_query: str = Field(description="The initial search query to use against the vector database.")

class ReflectorOutput(BaseModel):
    reflection: str = Field(description="Analysis of whether the retrieved evidence is sufficient to answer the user query.")
    sufficient: bool = Field(description="True if evidence is sufficient, False otherwise.")
    next_search_query: str = Field(description="If not sufficient, provide a new alternative search query to find missing information. Otherwise, leave empty.")

class AgentNodes:
    def __init__(self, vectorstore):
        self.vs = vectorstore
        # Initialize Groq client via LangChain for structured outputs
        api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(model="qwen/qwen3.6-27b", temperature=0.0, api_key=api_key)
        self.planner_llm = self.llm.with_structured_output(PlannerOutput)
        self.reflector_llm = self.llm.with_structured_output(ReflectorOutput)
        
    def plan_node(self, state: AgentState):
        query = state["query"]
        sys_prompt = "You are an expert research planner. Analyze the user's query and formulate a plan to answer it. Identify if this is a factual question, comparison, summary, or multi-document question. Provide the best initial search query to retrieve information from a vector database."
        msg = [SystemMessage(content=sys_prompt), HumanMessage(content=query)]
        
        try:
            result = self.planner_llm.invoke(msg)
            return {
                "plan": result.plan,
                "search_queries": [result.search_query],
                "retrieval_iterations": 0
            }
        except Exception as e:
            # Fallback if structured output fails
            return {
                "plan": "Directly search the database for the exact query.",
                "search_queries": [query],
                "retrieval_iterations": 0
            }

    def retrieve_node(self, state: AgentState):
        queries = state.get("search_queries", [])
        current_query = queries[-1] if queries else state["query"]
        selected_files = state.get("selected_files", [])
        
        # Retrieve chunks from VectorStore
        retrieved_chunks = self.vs.retrieve(current_query, top_k=4, selected_files=selected_files)
        
        # Avoid duplicate exact chunks in state
        existing_ids = set()
        for chunk in state.get("evidence", []):
            if "id" in chunk:
                existing_ids.add(chunk["id"])
                
        new_evidence = []
        for chunk in retrieved_chunks:
            if chunk["id"] not in existing_ids:
                new_evidence.append(chunk)
                
        return {
            "evidence": new_evidence,
            "retrieval_iterations": state.get("retrieval_iterations", 0) + 1
        }
        
    def reflect_node(self, state: AgentState):
        query = state["query"]
        evidence = state.get("evidence", [])
        iterations = state.get("retrieval_iterations", 0)
        
        if iterations >= 3:
            # Max iterations reached to prevent infinite loops
            return {
                "reflection": "Maximum retrieval iterations reached. Proceeding to generation with available evidence.",
                "sufficient_evidence": True
            }
            
        context_parts = []
        for i, chunk in enumerate(evidence):
             source = chunk.get("metadata", {}).get("filename", f"Doc {i}")
             context_parts.append(f"--- Context {i+1} (Source: {source}) ---\n{chunk.get('document', '')}")
        context_text = "\n".join(context_parts)
        
        if not context_text.strip():
            context_text = "No evidence retrieved yet."
            
        sys_prompt = f"""You are a reflection agent. Your task is to evaluate if the retrieved evidence is sufficient to answer the user query fully.
User Query: {query}

Retrieved Evidence:
{context_text}

Analyze the evidence. If it lacks critical information to answer the query, set sufficient=False and provide a next_search_query to find the missing details.
If the evidence is sufficient, set sufficient=True and leave next_search_query empty.
"""
        try:
            result = self.reflector_llm.invoke([SystemMessage(content=sys_prompt)])
            return {
                "reflection": result.reflection,
                "sufficient_evidence": result.sufficient,
                "search_queries": [result.next_search_query] if (not result.sufficient and result.next_search_query) else []
            }
        except Exception:
            # Fallback
            return {
                "reflection": "Error in reflection, proceeding to generation.",
                "sufficient_evidence": True
            }
