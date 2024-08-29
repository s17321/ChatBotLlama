import os
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.llama_cpp.llama_utils import (
    messages_to_prompt,
    completion_to_prompt,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from config import args

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
