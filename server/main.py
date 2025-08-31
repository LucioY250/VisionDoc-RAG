# En main.py

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
from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from logger import logger

# ==============================================================================
# GESTOR DE CICLO DE VIDA (LIFESPAN)
# ==============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Código de Arranque (STARTUP) ---
    logger.info("Iniciando la aplicación y cargando modelos base...")
    
    # Sincronizamos con el modelo usado en la ingestión
    app.state.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        app.state.vectorstore = Chroma(persist_directory=str(PERSIST_DIR), embedding_function=app.state.embeddings)
        app.state.chain = get_llm_chain(app.state.vectorstore) # <<< SIMPLIFICADO
        logger.info("Vectorstore existente cargado y cadena de RAG inicializada.")
    else:
        app.state.vectorstore = None
        app.state.chain = None
        logger.warning("No se encontró un vectorstore. El sistema está en espera hasta la subida de un documento.")

    logger.info("¡Aplicación lista para recibir peticiones!")
    
    yield
    
    # --- Código de Apagado (SHUTDOWN) ---
    logger.info("La aplicación se está apagando.")

# ==============================================================================
# CONFIGURACIÓN DE LA APLICACIÓN FASTAPI
# ==============================================================================
app = FastAPI(title="VisionDoc-RAG", lifespan=lifespan)

SERVER_ROOT = Path(__file__).parent
app.mount("/static", StaticFiles(directory=SERVER_ROOT / "static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.middleware("http")
async def catch_exception_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("UNHANDLED EXCEPTION")
        return JSONResponse(status_code=500, content={"error": str(exc)})

# ==============================================================================
# ENDPOINTS DE LA API
# ==============================================================================
@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        return JSONResponse(status_code=400, content={"error": "No se subieron archivos."})
    try:
        logger.info(f"Recibidos {len(files)} archivos para procesar en segundo plano.")
        
        # El pipeline de ingestión ahora solo devuelve el vectorstore.
        new_vectorstore = await run_in_threadpool(load_vectorstore, files)
        
        # Actualizamos el estado de la aplicación.
        app.state.vectorstore = new_vectorstore
        app.state.chain = get_llm_chain(app.state.vectorstore) # <<< SIMPLIFICADO
        
        logger.info("Vectorstore recargado y cadena actualizada tras la ingestión.")
        return {"message": "Archivos procesados y vectorstore actualizado."}
    except Exception as e:
        logger.exception("Error durante la subida de PDF")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    if not app.state.chain:
        return JSONResponse(status_code=400, content={"error": "El sistema no está listo. Por favor, suba documentos primero."})
    try:
        logger.info(f"Consulta de usuario: {question}")
        result = await run_in_threadpool(query_chain, app.state.chain, question)
        logger.info("Consulta exitosa.")
        return result
    except Exception as e:
        logger.exception("Error procesando la pregunta")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/test")
async def test():
    return {"message": "Testing successful..."}