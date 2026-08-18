import streamlit as st
import os

from ingestion import extract_text
from chunking import recursive_character_text_split
from vectorstore import VectorStore
from llm import LLMClient
from agent.graph import create_agent_graph
from langchain_core.messages import HumanMessage

# Initialize Streamlit Page
st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

# Initialize session state for vector store and LLM
if "vs" not in st.session_state:
    st.session_state.vs = VectorStore()
    
if "llm" not in st.session_state:
    st.session_state.llm = LLMClient()
    
if "agent_graph" not in st.session_state:
    st.session_state.agent_graph = create_agent_graph(st.session_state.vs)
    
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar: Document Management ---
with st.sidebar:
    st.title("📚 Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "txt", "docx"], 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    filename = uploaded_file.name
                    ext = filename.split(".")[-1].lower()
                    
                    # 1. Extract Text
                    text = extract_text(uploaded_file, ext)
                    
                    if text.startswith("ERROR") or text.startswith("WARNING"):
                        st.error(f"{filename}: {text}")
                        continue
                        
                    # 2. Chunk Text
                    chunks = recursive_character_text_split(text, chunk_size=500, chunk_overlap=50)
                    
                    # 3. Embed & Store
                    success, msg = st.session_state.vs.add_documents(filename, text, chunks)
                    if success:
                        st.success(msg)
                    else:
                        st.warning(msg)
        else:
            st.warning("Please upload files first.")
            
    st.divider()
    
    st.subheader("Indexed Files")
    indexed_files = st.session_state.vs.get_indexed_files()
    
    selected_files = []
    if indexed_files:
        for f in indexed_files:
            st.text(f"📄 {f}")
        st.divider()
        selected_files = st.multiselect(
            "Select files to query (leave empty to search all):",
            options=indexed_files,
            default=[]
        )
    else:
        st.info("No files currently indexed.")
        
    st.divider()
        
    if st.button("Clear Vector Store"):
        st.session_state.vs.clear_vector_store()
        st.success("Vector store cleared!")
        st.rerun()

# --- Main Area: Chat Interface ---
st.title("RAG Chatbot")
st.markdown("Ask questions based on your uploaded documents.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display Agent Workflow from history
        if "workflow" in message and message["workflow"]:
            with st.expander("Agent Workflow"):
                for w_step in message["workflow"]:
                    st.markdown(w_step)
                    
        # Display citations from history
        if "citations" in message and message["citations"]:
            with st.expander(f"Sources ({len(message['citations'])} chunks retrieved)"):
                for i, citation in enumerate(message["citations"]):
                    st.markdown(f"**Source {i+1}** ({citation['metadata']['filename']}):\n> {citation['document']}")

# Accept user input
if prompt := st.chat_input("Ask a question..."):
    # 1. Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Run the Agentic Workflow
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        workflow_steps = []
        
        # Initialize the state for the graph
        initial_state = {
            "messages": [HumanMessage(content=prompt)],
            "query": prompt,
            "search_queries": [],
            "evidence": [],
            "retrieval_iterations": 0,
            "selected_files": selected_files
        }
        
        retrieved_chunks = []
        for step in st.session_state.agent_graph.stream(initial_state, stream_mode="updates"):
            # Update the UI with intermediate agent thoughts
            if "planner" in step:
                msg = f"🤔 **Planning:** {step['planner'].get('plan', 'Deciding on approach...')}"
                status_placeholder.info(msg)
                workflow_steps.append(msg)
            elif "retriever" in step:
                iters = step['retriever'].get('retrieval_iterations', 1)
                msg = f"🔍 **Retrieval Cycle {iters}:** Fetching evidence from the database..."
                status_placeholder.info(msg)
                workflow_steps.append(msg)
                # Accumulate retrieved evidence
                retrieved_chunks.extend(step['retriever'].get('evidence', []))
            elif "reflector" in step:
                msg = f"🧠 **Reflecting on Evidence:** {step['reflector'].get('reflection', 'Analyzing information...')}"
                status_placeholder.info(msg)
                workflow_steps.append(msg)
                
        status_placeholder.empty() # Clear intermediate thoughts
        
        # 3. Generate Final Response using collected evidence
        response_placeholder = st.empty()
        full_response = ""
        
        # Call the LLM with streaming, passing the evidence collected by the agent
        stream = st.session_state.llm.generate_streaming_response(
            query=prompt,
            context_chunks=retrieved_chunks,
            chat_history=st.session_state.messages[:-1] # Exclude the current prompt
        )
        
        # Iterate over the stream
        import re
        for chunk in stream:
            if isinstance(chunk, str):
                full_response += chunk
            else:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    
            # Filter out <think> blocks for clean display
            display_response = re.sub(r'<think>.*?</think>\n*', '', full_response, flags=re.DOTALL)
            display_response = re.sub(r'<think>.*', '', display_response, flags=re.DOTALL)
            
            # Update the placeholder
            response_placeholder.markdown(display_response.strip() + " ▌")
            
        # Final update
        display_response = re.sub(r'<think>.*?</think>\n*', '', full_response, flags=re.DOTALL)
        display_response = re.sub(r'<think>.*', '', display_response, flags=re.DOTALL)
        response_placeholder.markdown(display_response.strip())
        
        # Extract <think> block for the workflow if it exists
        think_match = re.search(r'<think>(.*?)</think>', full_response, flags=re.DOTALL)
        if think_match:
            think_content = think_match.group(1).strip()
            if think_content:
                workflow_steps.append(f"🤖 **Model Reasoning:**\n```text\n{think_content}\n```")
        
        # Display Agent Workflow
        if workflow_steps:
            with st.expander("Agent Workflow"):
                for w_step in workflow_steps:
                    st.markdown(w_step)
        
        # Display citations
        if retrieved_chunks:
            with st.expander(f"Sources ({len(retrieved_chunks)} chunks retrieved)"):
                for i, citation in enumerate(retrieved_chunks):
                    st.markdown(f"**Source {i+1}** ({citation['metadata']['filename']}):\n> {citation['document']}")
                    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": display_response.strip(),
        "citations": retrieved_chunks,
        "workflow": workflow_steps
    })
