import streamlit as st
import os

from ingestion import extract_text
from chunking import recursive_character_text_split
from vectorstore import VectorStore
from llm import LLMClient

# Initialize Streamlit Page
st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

# Initialize session state for vector store and LLM
if "vs" not in st.session_state:
    st.session_state.vs = VectorStore()
    
if "llm" not in st.session_state:
    st.session_state.llm = LLMClient()
    
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
    if indexed_files:
        for f in indexed_files:
            st.text(f"📄 {f}")
    else:
        st.info("No files currently indexed.")
        
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
        # Optionally display citations if they exist in the message
        if "citations" in message and message["citations"]:
            with st.expander("Sources"):
                for i, citation in enumerate(message["citations"]):
                    st.markdown(f"**Source {i+1}** ({citation['metadata']['filename']}):\n> {citation['document']}")

# Accept user input
if prompt := st.chat_input("Ask a question..."):
    # 1. Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Retrieve relevant context
    retrieved_chunks = st.session_state.vs.retrieve(prompt, top_k=4)
    
    # 3. Generate response
    with st.chat_message("assistant"):
        # Create an empty placeholder to stream the response
        response_placeholder = st.empty()
        full_response = ""
        
        # Call the LLM with streaming
        stream = st.session_state.llm.generate_streaming_response(
            query=prompt,
            context_chunks=retrieved_chunks,
            chat_history=st.session_state.messages[:-1] # Exclude the current prompt
        )
        
        # Iterate over the stream
        for chunk in stream:
            # Check if it's a groq chunk object or an error string
            if isinstance(chunk, str):
                full_response += chunk
            else:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
            # Update the placeholder
            response_placeholder.markdown(full_response + "▌")
            
        # Final update
        response_placeholder.markdown(full_response)
        
        # Display citations
        if retrieved_chunks:
            with st.expander("Sources"):
                for i, citation in enumerate(retrieved_chunks):
                    st.markdown(f"**Source {i+1}** ({citation['metadata']['filename']}):\n> {citation['document']}")
                    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "citations": retrieved_chunks
    })
