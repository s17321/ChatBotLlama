from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from routers import llm_router

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

# Add the router to the FastAPI app
app.include_router(llm_router)

# Run the FastAPI app if this file is executed
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
