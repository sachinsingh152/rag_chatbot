import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables (e.g. GROQ_API_KEY)
load_dotenv()

class LLMClient:
    def __init__(self, model_name: str = "qwen/qwen3.6-27b"):
        """
        Initializes the Groq client. Expects GROQ_API_KEY in the environment.
        Using llama-3.1-8b-instant as a fast and capable default model.
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not found. Please set it in your .env file.")
            
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        
    def generate_streaming_response(self, query: str, context_chunks: list[dict], chat_history: list[dict] = None):
        """
        Generates a streaming response from Groq using the provided context.
        """
        if chat_history is None:
            chat_history = []
            
        # 1. Build the Context String
        if not context_chunks:
            context_text = "No relevant context found in the documents."
        else:
            context_parts = []
            for i, chunk in enumerate(context_chunks):
                source = chunk.get("metadata", {}).get("filename", f"Document {i+1}")
                context_parts.append(f"--- Context {i+1} (Source: {source}) ---\n{chunk.get('document', '')}\n")
            context_text = "\n".join(context_parts)
            
        # 2. Build the System Prompt
        system_prompt = f"""You are a helpful and precise Retrieval-Augmented Generation (RAG) assistant.
Your task is to answer the user's question based strictly on the provided context below.

CONTEXT:
{context_text}

INSTRUCTIONS:
1. Answer the question using ONLY the provided context.
2. If the answer is not contained in the context, say exactly: "I don't know based on the provided documents." Do not guess or use outside knowledge.
3. If the context contains the answer, be concise and clear.
4. EVERY claim in your answer MUST include an inline citation indicating the source document name, formatted exactly as [Source: filename].
5. Output ONLY the final answer. DO NOT output any <think> tags, reasoning steps, or internal monologue.
"""

        # 3. Construct the Message History
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history (to maintain conversation flow)
        for msg in chat_history:
            if msg["role"] in ["user", "assistant"]:
                # Exclude any old context or raw chunks from history, just the conversational text
                messages.append({"role": msg["role"], "content": msg["content"]})
                
        # Add the current user query
        messages.append({"role": "user", "content": query})
        
        # 4. Call Groq API with Streaming
        try:
            stream = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=0.0, # Low temperature for more deterministic/factual answers
                max_tokens=4096,
                stream=True,
            )
            return stream
        except Exception as e:
            error_message = str(e)
            # Return a simple generator that yields the error so the UI doesn't crash entirely
            def error_stream():
                yield f"ERROR: Failed to call Groq API. Details: {error_message}"
            return error_stream()
