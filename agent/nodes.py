import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from .state import AgentState

from .schemas import PlannerOutput, ReflectorOutput, HallucinationOutput
from .prompts import PLANNER_PROMPT, REFLECTOR_PROMPT, GENERATOR_PROMPT, HALLUCINATION_PROMPT

class AgentNodes:
    def __init__(self, vectorstore):
        self.vs = vectorstore
        # Initialize Gemini client via LangChain for structured outputs
        api_key = os.getenv("GEMINI_API_KEY")
        
        # Use gemini-3.5-flash for high quality generation and gemini-3.5-flash-lite for fast routing
        self.llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.0, max_output_tokens=4000, max_retries=0, api_key=api_key)
        self.structured_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0, max_output_tokens=4000, max_retries=0, api_key=api_key)
        
        self.planner_llm = self.structured_llm.with_structured_output(PlannerOutput)
        self.reflector_llm = self.structured_llm.with_structured_output(ReflectorOutput)
        self.hallucination_llm = self.structured_llm.with_structured_output(HallucinationOutput)
        
    def plan_node(self, state: AgentState):
        query = state["query"]
        sys_prompt = PLANNER_PROMPT
        msg = [SystemMessage(content=sys_prompt)]
        for m in state.get("messages", [])[-11:-1]:
            if m.type in ["human", "ai", "user", "assistant"]:
                msg.append(m)
        msg.append(HumanMessage(content=query))
        
        try:
            print(f"[DEBUG] plan_node: Calling Planner LLM...")
            result = self.planner_llm.invoke(msg)
            print(f"[DEBUG] plan_node: Planner LLM finished.")
            return {
                "plan": result.plan,
                "search_queries": result.search_queries,
                "retrieval_iterations": 0
            }
        except Exception as e:
            print(f"[ERROR] plan_node Exception: {e}")
            return {
                "plan": "Error in planning",
                "search_queries": [query],
                "retrieval_iterations": 0
            }

    def retrieve_node(self, state: AgentState):
        queries = state.get("search_queries", [])
        iters = state.get("retrieval_iterations", 0)
        
        if iters == 0:
            current_queries = queries[-3:] if len(queries) >= 3 else queries
        else:
            current_queries = [queries[-1]] if queries else []
            
        selected_files = state.get("selected_files", [])
        
        if not selected_files:
            return {
                "evidence": [],
                "retrieval_iterations": iters + 1
            }
        
        # Avoid duplicate exact chunks in state
        existing_ids = set()
        for chunk in state.get("evidence", []):
            if "id" in chunk:
                existing_ids.add(chunk["id"])
                
        new_evidence = []
        for q in current_queries:
            retrieved_chunks = self.vs.retrieve(q, top_k=4, selected_files=selected_files, username=state.get("username"))
            for chunk in retrieved_chunks:
                if chunk["id"] not in existing_ids:
                    existing_ids.add(chunk["id"])
                    new_evidence.append(chunk)
                    
        return {
            "evidence": new_evidence,
            "retrieval_iterations": iters + 1
        }

    def web_search_node(self, state: AgentState):
        queries = state.get("search_queries", [])
        current_query = queries[-1] if queries else state["query"]
        
        try:
            import requests
            import urllib.parse
            query_encoded = urllib.parse.quote(current_query)
            headers = {'User-Agent': 'RAG-Agent (rag@example.com)'}
            search_url = f'https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query_encoded}&utf8=&format=json'
            res = requests.get(search_url, headers=headers, timeout=5).json()
            
            web_evidence = []
            if res.get('query', {}).get('search'):
                # Take top 2 Wikipedia results for speed
                for i, result in enumerate(res['query']['search'][:2]):
                    title = result['title']
                    title_encoded = urllib.parse.quote(title)
                    page_url = f'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&titles={title_encoded}&format=json'
                    page_res = requests.get(page_url, headers=headers, timeout=5).json()
                    pages = page_res.get('query', {}).get('pages', {})
                    for page_id, page_data in pages.items():
                        if 'extract' in page_data:
                            web_evidence.append({
                                "id": f"wiki_{i}",
                                "document": page_data['extract'][:3000], # Keep chunks concise but long enough for answers
                                "metadata": {"filename": f"Wikipedia: {title}", "url": f"https://en.wikipedia.org/wiki/{title_encoded}"}
                            })
                            
            if not web_evidence:
                raise Exception("No Wikipedia results found for this query.")
        except Exception as e:
            print(f"Web Search Error: {e}")
            web_evidence = [{
                "id": "web_error",
                "document": f"The internet search service is temporarily unavailable or returned no results (Error: {str(e)}).",
                "metadata": {"filename": "System Status", "url": "N/A"}
            }]
            
        return {
            "web_evidence": web_evidence,
            "retrieval_iterations": state.get("retrieval_iterations", 0) + 1
        }
        
    def reflect_node(self, state: AgentState):
        query = state["query"]
        evidence = state.get("evidence", []) + state.get("web_evidence", [])
        iterations = state.get("retrieval_iterations", 0)
        selected_files = state.get("selected_files", [])
        
        if iterations >= 1:
            print(f"[DEBUG] reflect_node: Fast-path skip (max iterations reached).")
            return {
                "reflection": "Maximum retrieval iterations reached. Proceeding to generation with available evidence.",
                "sufficient_evidence": True,
                "use_web_search": False
            }
            
        if any(chunk.get("metadata", {}).get("filename") == "System Status" for chunk in evidence):
            return {
                "reflection": "Web search service encountered an error. Proceeding to generation to inform user.",
                "sufficient_evidence": True,
                "use_web_search": False
            }
            
        if not selected_files and not state.get("web_evidence"):
            return {
                "reflection": "No local documents selected. Falling back to live internet search.",
                "sufficient_evidence": False,
                "use_web_search": True
            }
            
        context_parts = []
        for i, chunk in enumerate(evidence):
             source = chunk.get("metadata", {}).get("filename", f"Doc {i}")
             context_parts.append(f"--- Context {i+1} (Source: {source}) ---\n{chunk.get('document', '')}")
        context_text = "\n".join(context_parts)
        
        if not context_text.strip():
            context_text = "No evidence retrieved yet."
            
        sys_prompt = REFLECTOR_PROMPT.format(query=query, context_text=context_text)
        try:
            msg = [SystemMessage(content=sys_prompt)]
            for m in state.get("messages", [])[-11:-1]:
                if m.type in ["human", "ai", "user", "assistant"]:
                    msg.append(m)
            msg.append(HumanMessage(content=f"Please evaluate the evidence for the query: '{query}'"))
            print(f"[DEBUG] reflect_node: Calling Reflector LLM...")
            result = self.reflector_llm.invoke(msg)
            print(f"[DEBUG] reflect_node: Reflector LLM finished.")
            return {
                "reflection": result.reflection,
                "sufficient_evidence": result.sufficient,
                "use_web_search": result.use_web_search,
                "search_queries": [result.next_search_query] if (not result.sufficient and result.next_search_query) else []
            }
        except Exception as e:
            print(f"[ERROR] reflect_node Exception: {e}")
            return {
                "reflection": "Error in reflection, proceeding to generation.",
                "sufficient_evidence": True,
                "use_web_search": False
            }

    def generate_node(self, state: AgentState):
        query = state["query"]
        evidence = state.get("evidence", []) + state.get("web_evidence", [])
        hallucination_feedback = state.get("hallucination_feedback", "")
        
        context_parts = []
        for i, chunk in enumerate(evidence):
            source = chunk.get("metadata", {}).get("filename", f"Document {i+1}")
            context_parts.append(f"--- Context {i+1} (Source: {source}) ---\n{chunk.get('document', '')}")
        context_text = "\n".join(context_parts)
        
        sys_prompt = GENERATOR_PROMPT.format(context_text=context_text)
        if hallucination_feedback:
            sys_prompt += f"\n\nPREVIOUS ATTEMPT FEEDBACK: {hallucination_feedback}\nPlease correct your answer to avoid hallucinations."

        messages = [{"role": "system", "content": sys_prompt}]
        for msg in state["messages"][-11:-1]:
            if msg.type in ["human", "ai", "user", "assistant"]:
                role = "user" if msg.type in ["human", "user"] else "assistant"
                messages.append({"role": role, "content": msg.content})
        messages.append({"role": "user", "content": query})
        
        print(f"[DEBUG] generate_node: Calling Generator LLM with {len(messages)} messages...")
        result = self.llm.invoke(messages)
        print(f"[DEBUG] generate_node: Generator LLM finished.")
        
        # Ensure content is a string (Gemini sometimes returns a list of parts)
        content = result.content
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict) and "text" in part)
        elif not isinstance(content, str):
            content = str(content)
            
        return {
            "draft_response": content,
            "generation_iterations": state.get("generation_iterations", 0) + 1
        }

    def hallucination_node(self, state: AgentState):
        draft = state.get("draft_response", "")
        evidence = state.get("evidence", []) + state.get("web_evidence", [])
        
        context_parts = []
        for i, chunk in enumerate(evidence):
            context_parts.append(f"{chunk.get('document', '')}")
        context_text = "\n".join(context_parts)
        
        sys_prompt = HALLUCINATION_PROMPT.format(context_text=context_text, draft=draft)
        try:
            msg = [SystemMessage(content=sys_prompt)]
            for m in state.get("messages", [])[-11:-1]:
                if m.type in ["human", "ai", "user", "assistant"]:
                    msg.append(m)
            msg.append(HumanMessage(content="Please evaluate the draft response against the evidence and chat history."))
            print(f"[DEBUG] hallucination_node: Calling Hallucination LLM...")
            result = self.hallucination_llm.invoke(msg)
            print(f"[DEBUG] hallucination_node: Hallucination LLM finished.")
            if result.is_hallucinated:
                return {
                    "hallucination_feedback": result.feedback
                }
            else:
                return {
                    "final_response": draft,
                    "hallucination_feedback": ""
                }
        except Exception as e:
            print(f"[ERROR] hallucination_node Exception: {e}")
            return {
                "final_response": draft,
                "hallucination_feedback": ""
            }
