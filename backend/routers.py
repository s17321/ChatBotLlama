from fastapi import APIRouter
from utils import vector_store, llm
from models import Document, LLMResponse
from memory import last_query, last_response

llm_router = APIRouter()

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
