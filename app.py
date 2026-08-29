import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Disable noisy ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from core.ingestion import extract_text
from core.chunking import recursive_character_text_split
from core.vectorstore import VectorStore
from agent.graph import create_agent_graph
from langchain_core.messages import HumanMessage
from core.auth import init_db, create_user, authenticate_user, load_chat_history, save_chat_message, clear_chat_history, get_user_conversations
import uuid

# Initialize Streamlit Page
st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

# Initialize database
init_db()

# Initialize session state for vector store and LLM
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Welcome to RAG Chatbot")
    st.markdown("Please log in or sign up to continue.")
    
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_user")
        login_password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if authenticate_user(login_username, login_password):
                st.session_state.authenticated = True
                st.session_state.username = login_username
                
                # Fetch conversations or create a new one
                user_convs = get_user_conversations(login_username)
                if user_convs:
                    st.session_state.conversation_id = user_convs[0]['id']
                else:
                    st.session_state.conversation_id = str(uuid.uuid4())
                    
                st.session_state.messages = load_chat_history(login_username, st.session_state.conversation_id)
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
                
    with tab2:
        st.subheader("Sign Up")
        signup_username = st.text_input("New Username", key="signup_user")
        signup_password = st.text_input("New Password", type="password", key="signup_pass")
        signup_confirm = st.text_input("Confirm Password", type="password", key="signup_confirm")
        if st.button("Sign Up"):
            if not signup_username or not signup_password:
                st.error("Please fill in all fields.")
            elif signup_password != signup_confirm:
                st.error("Passwords do not match.")
            else:
                if create_user(signup_username, signup_password):
                    st.success("Account created successfully! You can now log in.")
                else:
                    st.error("Username already exists. Please choose a different one.")
                    
    st.stop()  # Stop execution until authenticated

if "vs" not in st.session_state:
    st.session_state.vs = VectorStore()
    
# Note: llm is now managed internally by the agent_graph using Gemini
    
if "agent_graph" not in st.session_state:
    pass # we will create the graph per request to prevent event loop deadlocks
    
if "messages" not in st.session_state:
    if "username" in st.session_state:
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = load_chat_history(st.session_state.username, st.session_state.conversation_id)
    else:
        st.session_state.messages = []

# --- Sidebar: Document Management ---
with st.sidebar:
    st.markdown(f"**Logged in as:** {st.session_state.get('username', 'Unknown')}")
    if st.button("Logout"):
        st.session_state.authenticated = False
        if "username" in st.session_state:
            del st.session_state.username
        if "messages" in st.session_state:
            del st.session_state.messages
        if "conversation_id" in st.session_state:
            del st.session_state.conversation_id
        st.rerun()
        
    st.divider()
    
    # --- Chat Threads ---
    st.subheader("💬 Conversations")
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.conversation_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
        
    user_convs = get_user_conversations(st.session_state.get('username', ''))
    for conv in user_convs:
        # Highlight active conversation
        btn_type = "primary" if conv['id'] == st.session_state.get('conversation_id') else "secondary"
        if st.button(f"📄 {conv['title']}", key=f"conv_{conv['id']}", type=btn_type, use_container_width=True):
            st.session_state.conversation_id = conv['id']
            st.session_state.messages = load_chat_history(st.session_state.username, conv['id'])
            st.rerun()
        
    st.divider()
    
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
                    success, msg = st.session_state.vs.add_documents(filename, text, chunks, st.session_state.username)
                    if success:
                        st.success(msg)
                    else:
                        st.warning(msg)
        else:
            st.warning("Please upload files first.")
            
    st.divider()
    
    st.subheader("Indexed Files")
    indexed_files = st.session_state.vs.get_indexed_files(st.session_state.username)
    
    selected_files = []
    if indexed_files:
        for f in indexed_files:
            st.text(f"📄 {f}")
        st.divider()
        selected_files = st.multiselect(
            "Select files to query (leave empty for web search only):",
            options=indexed_files,
            default=[]
        )
    else:
        st.info("No files currently indexed.")
        
    st.divider()
        
    if st.button("Clear Vector Store"):
        st.session_state.vs.clear_vector_store(st.session_state.username)
        st.success("Vector store cleared!")
        st.rerun()
        
    if st.button("Clear Chat History"):
        clear_chat_history(st.session_state.username, st.session_state.conversation_id)
        st.session_state.messages = []
        st.success("Chat history cleared!")
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

if "processing" not in st.session_state:
    st.session_state.processing = False
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = None

# Accept user input
prompt_input = st.chat_input("Ask a question...", disabled=st.session_state.processing)

if prompt_input:
    st.session_state.current_prompt = prompt_input
    st.session_state.processing = True
    st.rerun()

if st.session_state.processing and st.session_state.current_prompt:
    prompt = st.session_state.current_prompt
    
    # 1. Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Add user message to chat history
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    save_chat_message(st.session_state.username, st.session_state.conversation_id, user_msg)
    
    # 2. Run the Agentic Workflow
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        workflow_steps = []
        
        from langchain_core.messages import HumanMessage, AIMessage
        
        chat_history_messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_history_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history_messages.append(AIMessage(content=msg["content"]))
                
        # Initialize the state for the graph
        initial_state = {
            "messages": chat_history_messages,
            "query": prompt,
            "search_queries": [],
            "evidence": [],
            "web_evidence": [],
            "retrieval_iterations": 0,
            "selected_files": selected_files,
            "hallucination_feedback": "",
            "username": st.session_state.username
        }
        
        retrieved_chunks = []
        final_response = ""
        
        import time
        from agent.graph import create_agent_graph
        agent_graph = create_agent_graph(st.session_state.vs)
        
        try:
            for step in agent_graph.stream(initial_state, stream_mode="updates"):
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
                    retrieved_chunks.extend(step['retriever'].get('evidence', []))
                elif "web_search" in step:
                    msg = f"🌐 **Web Search:** Falling back to live internet search..."
                    status_placeholder.info(msg)
                    workflow_steps.append(msg)
                    retrieved_chunks.extend(step['web_search'].get('web_evidence', []))
                elif "reflector" in step:
                    msg = f"🧠 **Reflecting on Evidence:** {step['reflector'].get('reflection', 'Analyzing information...')}"
                    status_placeholder.info(msg)
                    workflow_steps.append(msg)
                elif "generator" in step:
                    msg = f"✍️ **Drafting:** Writing response based strictly on evidence..."
                    status_placeholder.info(msg)
                    workflow_steps.append(msg)
                elif "hallucination_checker" in step:
                    checker_output = step["hallucination_checker"]
                    if checker_output.get("hallucination_feedback"):
                        msg = f"❌ **Hallucination Detected:** Found unsupported claims. Rewriting..."
                    else:
                        msg = f"✅ **Fact Check Passed:** Response is fully supported by evidence."
                        final_response = checker_output.get("final_response", "")
                    status_placeholder.info(msg)
                    workflow_steps.append(msg)
        except Exception as e:
            st.error(f"⚠️ **Agent encountered an error:** {str(e)}")
            final_response = "I encountered an error while processing your request. Please try again."
            
        status_placeholder.empty() # Clear intermediate thoughts
        
        # 3. Display Final Response with simulated streaming
        response_placeholder = st.empty()
        
        import re
        # Filter out <think> blocks or 'Thinking Process:' for clean display
        display_response = re.sub(r'<think>.*?</think>\n*', '', final_response, flags=re.DOTALL)
        display_response = re.sub(r'<think>.*', '', display_response, flags=re.DOTALL)
        display_response = re.sub(r'Thinking Process:.*?Output:', '', display_response, flags=re.DOTALL | re.IGNORECASE)
        display_response = re.sub(r'Thinking Process:.*?$', '', display_response, flags=re.DOTALL | re.IGNORECASE)
        
        # Simulate typing effect to make UI feel responsive
        current_display = ""
        words = display_response.split(" ")
        for word in words:
            current_display += word + " "
            response_placeholder.markdown(current_display + " ▌")
            import time
            time.sleep(0.015) # 15ms per word
            
        response_placeholder.markdown(current_display.strip())
        
        # Extract <think> block for the workflow if it exists
        think_match = re.search(r'<think>(.*?)</think>', final_response, flags=re.DOTALL)
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
    assistant_msg = {
        "role": "assistant", 
        "content": display_response.strip(),
        "citations": retrieved_chunks,
        "workflow": workflow_steps
    }
    st.session_state.messages.append(assistant_msg)
    save_chat_message(st.session_state.username, st.session_state.conversation_id, assistant_msg)
    
    st.session_state.processing = False
    st.session_state.current_prompt = None
    st.rerun()
