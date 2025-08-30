# En modules/query_handlers.py
import os
from logger import logger
from pathlib import Path

# Usamos la ruta robusta definida en load_vectorstore
SERVER_ROOT = Path(__file__).parent.parent
BASE_URL = "http://127.0.0.1:8000"

def user_wants_image(user_input: str) -> bool:
    """
    Determina si la intención del usuario es ver una imagen usando palabras clave.
    """
    keywords = ["muéstrame", "enséñame", "diagrama", "imagen", "gráfico", "arquitectura", "flujo", "cómo se ve"]
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in keywords)

def query_chain(chain, user_input: str):
    """
    Ejecuta la cadena de RAG y formatea la respuesta, incluyendo la URL de una imagen
    solo si la intención del usuario es visual.
    """
    try:
        logger.debug(f"Ejecutando la cadena para la entrada: {user_input}")
        result = chain.invoke({"query": user_input})
        
        image_url = None
        source_documents = result.get("source_documents", [])

        ### MEJORA CRÍTICA: Lógica de Visualización Inteligente ###
        if source_documents and user_wants_image(user_input):
            top_doc = source_documents[0]
            metadata = top_doc.metadata
            source_filename = metadata.get("source")
            page_number = metadata.get("page_number")

            if source_filename and page_number:
                filename_without_ext = os.path.splitext(source_filename)[0]
                image_filename = f"{filename_without_ext}_p{page_number}_full.png"
                
                # Construimos la URL completa y segura
                relative_path = os.path.join("static/images", image_filename).replace("\\", "/")
                image_url = f"{BASE_URL}/{relative_path}"
                logger.debug(f"Imagen relevante encontrada en la página {page_number}. URL: {image_url}")

        response = {
            "response": result["result"],
            "sources": [doc.metadata.get("source", "N/A") for doc in source_documents],
            "image_url": image_url
        }
        
        logger.debug(f"Respuesta de la cadena: {response}")
        return response
        
    except Exception as e:
        logger.exception("Error en query_chain")
        raise