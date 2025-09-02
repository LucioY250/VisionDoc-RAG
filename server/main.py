# main.py

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
from fastapi.concurrency import run_in_threadpool

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from modules.load_vectorstore import load_vectorstore, PERSIST_DIR
from modules.llm import get_rag_chain
from modules.query_handlers import query_chain
from logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    Loads resource-intensive models once to be reused across requests.
    """
    logger.info("Starting application and loading base models...")
    
    # Ensure embedding model consistency between ingestion and querying
    app.state.embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        app.state.vectorstore = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=app.state.embeddings)
        app.state.chain = get_rag_chain(app.state.vectorstore)
        logger.info("Existing vectorstore loaded and RAG chain initialized.")
    else:
        app.state.vectorstore, app.state.chain = None, None
        logger.warning("No vectorstore found. System is waiting for a document upload.")
    
    logger.info("Application ready to receive requests!")
    yield
    logger.info("Application is shutting down.")

app = FastAPI(title="VisionDoc-RAG", lifespan=lifespan)

# --- Static File Serving ---
SERVER_ROOT = Path(__file__).parent
app.mount("/static", StaticFiles(directory=SERVER_ROOT / "static"), name="static")

# --- Middleware Configuration ---
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def catch_exception_middleware(request: Request, call_next):
    """Global exception handler to prevent server crashes."""
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("UNHANDLED EXCEPTION")
        return JSONResponse(status_code=500, content={"error": str(exc)})

# --- API Endpoints ---
@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """Handles PDF file uploads and triggers the ingestion pipeline."""
    if not files:
        return JSONResponse(status_code=400, content={"error": "No files were uploaded."})
    try:
        logger.info(f"Received {len(files)} files for background processing.")
        # Run the heavy ingestion task in a separate thread to avoid blocking the server
        new_vectorstore = await run_in_threadpool(load_vectorstore, files)
        
        # Update the application state with the new data
        app.state.vectorstore = new_vectorstore
        app.state.chain = get_rag_chain(app.state.vectorstore)
        
        logger.info("Vectorstore reloaded and chain updated after ingestion.")
        return {"message": "Files processed and vectorstore updated."}
    except Exception as e:
        logger.exception("Error during PDF upload")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    """Handles user queries by invoking the RAG chain."""
    if not app.state.chain:
        return JSONResponse(status_code=400, content={"error": "The system is not ready. Please upload documents first."})
    try:
        logger.info(f"User query: {question}")
        result = await run_in_threadpool(query_chain, app.state.chain, question)
        logger.info("Query successful.")
        return result
    except Exception as e:
        logger.exception("Error processing question")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/test")
async def test():
    """A simple endpoint to check if the server is running."""
    return {"message": "Testing successful..."}