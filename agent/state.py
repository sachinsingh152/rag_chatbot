import operator
from typing import TypedDict, Annotated, Any, List
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Conversation history + current user message
    messages: Annotated[list[AnyMessage], add_messages]
    
    # Extracted current query
    query: str
    
    # Planner output
    plan: str
    
    # Retrieval tracking
    search_queries: Annotated[list[str], operator.add] # Append new queries
    evidence: Annotated[list[dict], operator.add] # Append new chunks
    
    # Reflection output
    reflection: str
    sufficient_evidence: bool
    retrieval_iterations: int
    selected_files: list[str]
    
    # Final Output
    final_response: str
