# En modules/query_handlers.py
import os
from pathlib import Path
from logger import logger

SERVER_ROOT = Path(__file__).parent.parent
BASE_URL = "http://127.0.0.1:8000"

def user_wants_image(user_input: str) -> bool:
    keywords = ["show me", "diagram", "image", "graph", "architecture", "flow", "what does it look like", "muéstrame", "enséñame", "diagrama", "imagen", "gráfico", "arquitectura", "flujo"]
    return any(keyword in user_input.lower() for keyword in keywords)

def query_chain(chain, user_input: str):
    """
    Orquesta el RAG. Ya no necesita lógica de dos etapas porque la ingestión es muy inteligente.
    """
    try:
        logger.debug(f"Executing High-Definition RAG for: {user_input}")
        result = chain.invoke({"query": user_input})
        
        image_url = None
        source_documents = result.get("source_documents", [])
        
        # Lógica de visualización simplificada: Si la respuesta es sobre una imagen,
        # la ingestión ya habrá guardado el archivo.
        # Por simplicidad para el reto, asumiremos que si el usuario quiere una imagen,
        # el retriever encontrará la página correcta.
        if source_documents and user_wants_image(user_input):
            top_doc = source_documents[0]
            metadata = top_doc.metadata
            source_filename = metadata.get("source")
            page_number = metadata.get("page_number")
            
            if source_filename and page_number:
                # Heurística para encontrar la imagen del diagrama en esa página
                # Esto se podría mejorar buscando el ID de la imagen en los metadatos
                # pero es una buena aproximación.
                image_filename_to_show = f"{os.path.splitext(source_filename)[0]}_p{page_number}_img0.png" # Asumimos que es la primera imagen
                
                relative_path = os.path.join("static/images", image_filename_to_show).replace("\\", "/")
                image_url = f"{BASE_URL}/{relative_path}"
                logger.debug(f"Displaying potential image: {image_filename_to_show}")

        return {
            "response": result["result"],
            "sources": [doc.metadata.get("source", "N/A") for doc in source_documents],
            "image_url": image_url
        }
    except Exception as e:
        logger.exception("Error in query_chain")
        raise