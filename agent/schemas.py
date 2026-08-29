from pydantic import BaseModel, Field

class PlannerOutput(BaseModel):
    plan: str = Field(default="", description="Step by step reasoning plan to answer the user's query.")
    search_queries: list[str] = Field(default_factory=list, description="Generate 3 varied search queries to retrieve information from a vector database.")

class ReflectorOutput(BaseModel):
    reflection: str = Field(default="No reflection provided.", description="Analysis of whether the retrieved evidence is sufficient to answer the user query.")
    sufficient: bool = Field(default=True, description="True if evidence is sufficient, False otherwise.")
    use_web_search: bool = Field(default=False, description="True if local database lacks the answer and web search is required.")
    next_search_query: str = Field(default="", description="If not sufficient, provide a new alternative search query. Otherwise, leave empty.")

class HallucinationOutput(BaseModel):
    is_hallucinated: bool = Field(default=False, description="True if the draft response contains facts not supported by the evidence.")
    feedback: str = Field(default="", description="If hallucinated, provide feedback on what to fix. Otherwise, leave empty.")
