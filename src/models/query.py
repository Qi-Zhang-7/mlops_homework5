from pydantic import BaseModel
from typing import List, Optional

class RAGRequest(BaseModel):
    question: str
    num_responses: int = 3
    # populate the rag response
    # add any additional metadata
class AnswerItem(BaseModel):
    text: str
    score: Optional[float] = None
    source: Optional[str] = None

class RAGResponse(BaseModel):
    answers: List [AnswerItem]
    
