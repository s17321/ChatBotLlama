from pydantic import BaseModel
from typing import List

# Define Pydantic model for the document context
class Document(BaseModel):
    page_content: str
    metadata: dict

    @classmethod
    def from_langchain_document(cls, doc):
        return cls(page_content=doc.page_content, metadata=doc.metadata)

# Define Pydantic model for the response
class LLMResponse(BaseModel):
    completion: str
    context: List[Document]
