from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import AgentNodes

def create_agent_graph(vectorstore):
    # Initialize the nodes with the vector store
    nodes = AgentNodes(vectorstore)
    
    # Initialize the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planner", nodes.plan_node)
    workflow.add_node("retriever", nodes.retrieve_node)
    workflow.add_node("web_search", nodes.web_search_node)
    workflow.add_node("reflector", nodes.reflect_node)
    workflow.add_node("generator", nodes.generate_node)
    workflow.add_node("hallucination_checker", nodes.hallucination_node)
    
    # Set the entry point
    workflow.set_entry_point("planner")
    
    # Define normal edges
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "reflector")
    workflow.add_edge("web_search", "reflector")
    workflow.add_edge("generator", "hallucination_checker")
    
    # Define conditional edges from reflector
    def reflection_router(state: AgentState):
        if state.get("sufficient_evidence", False):
            return "generate"
        if state.get("use_web_search", False):
            return "web_search"
        return "retrieve"
        
    workflow.add_conditional_edges(
        "reflector",
        reflection_router,
        {
            "generate": "generator",
            "web_search": "web_search",
            "retrieve": "retriever"
        }
    )
    
    # Define conditional edges from hallucination checker
    def hallucination_router(state: AgentState):
        if state.get("hallucination_feedback") and state.get("generation_iterations", 0) < 1:
            return "generate" # rewrite
        return "end"
        
    workflow.add_conditional_edges(
        "hallucination_checker",
        hallucination_router,
        {
            "generate": "generator",
            "end": END
        }
    )
    
    # Compile the graph
    return workflow.compile()
