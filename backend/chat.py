import argparse
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Embedding Llama chatbot with PDFs.')
parser.add_argument('--pdf_file', type=str, default='ipb4.pdf', help='The PDF file to load')
parser.add_argument('--temperature', type=float, default=0.1, help='The temperature for the LlamaCPP model')
parser.add_argument('--model_path', type=str, default='/Users/lukasz/Desktop/StoryNook/data/llamaModels/smaller.gguf', help='The path to the LlamaCPP model')
args = parser.parse_args()

# Initialize API router for the LLM endpoints
app = FastAPI(
    title="PDF Chat API",
    description="An API to chat with pdf data, based on Llama LLM.",
    version="1.0.0"
)

# Define CORS origins for local development
origins = [
    "http://localhost",
    "http://localhost:5173",
]

# Apply CORS middleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
llm_router = APIRouter()

# Load and split the specified PDF document
current_script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_script_dir, 'Data', args.pdf_file)
pdf_loader = PyPDFLoader(file_path, extract_images=True)
document_pages = pdf_loader.load_and_split()

# Initialize embedding function with SentenceTransformer
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a vector store from the document pages using the embedding function
vector_store = Chroma.from_documents(document_pages, embedding_function)

# Initialize the LlamaCPP model with command-line arguments
llm = LlamaCPP(
    model_path=args.model_path,
    temperature=args.temperature,
    max_new_tokens=256,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": -1},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)

# Global variables to store the last request and response
last_query = None
last_response = None

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

# Define the API endpoint for processing LLM input
@llm_router.post("/llm-input", response_model=LLMResponse)
async def process_llm_input(user_input: str):
    global last_query, last_response

    # Perform a similarity search on the vector store using the user input
    similar_documents = vector_store.similarity_search(user_input)
    similar_documents_pydantic = [Document.from_langchain_document(doc) for doc in similar_documents]

    # Construct the prompt with the history of the last interaction
    history_text = ""
    if last_query and last_response:
        history_text = f"User: {last_query}\nModel: {last_response}\n"

    prompt = f"You are a helpful model, an assistant that helps answer questions about university documents based on the provided context. If you don't know the answer you sa you don't know and do not create one out of imagination. Context from previous interaction:\n{history_text}\n. Based on this context (use it only if it is needed and applies for the new question) answer the new user query: {user_input}\n.Answer based on the previous context if needed and also use the following documents as your source of knowledge, select only the data that applies and makes sense for the new user query: {str(similar_documents_pydantic)}"
    
    # Generate a completion using the LLM
    completion = llm.complete(prompt)
    
    # Update the last query and response
    last_query = user_input
    last_response = completion.text
    
    # Return the generated text and context as the response
    return LLMResponse(completion=completion.text, context=similar_documents_pydantic)

# Add the router to the FastAPI app
app.include_router(llm_router)

# Run the FastAPI app if this file is executed
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
