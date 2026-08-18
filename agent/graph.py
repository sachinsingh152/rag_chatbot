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
    workflow.add_node("reflector", nodes.reflect_node)
    
    # Set the entry point
    workflow.set_entry_point("planner")
    
    # Define normal edges
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "reflector")
    
    # Define conditional edges from reflector
    def reflection_router(state: AgentState):
        if state.get("sufficient_evidence", False):
            return "end"
        return "retrieve"
        
    workflow.add_conditional_edges(
        "reflector",
        reflection_router,
        {
            "end": END,
            "retrieve": "retriever"
        }
    )
    
    # Compile the graph
    return workflow.compile()
