# En modules/query_handlers.py
import os
from pathlib import Path
from logger import logger
import fitz
import replicate
from langchain_groq import ChatGroq
from modules.load_vectorstore import UPLOAD_DIR

# --- CONFIGURACIÓN ---
SERVER_ROOT = Path(__file__).parent.parent
BASE_URL = "http://127.0.0.1:8000"
groq_llm = ChatGroq(groq_api_key=os.environ.get("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile", temperature=0)

# --- FUNCIONES DE LA LÓGICA DE DOS ETAPAS ---
def user_wants_image(user_input: str) -> bool:
    """Determina si la intención del usuario es visual."""
    keywords = ["muéstrame", "enséñame", "diagrama", "imagen", "gráfico", "arquitectura", "flujo", "cómo se ve"]
    return any(keyword in user_input.lower() for keyword in keywords)

def describe_image_on_demand(image_path: str) -> str:
    """Llama al Agente Especialista Visual (VLM) solo cuando es necesario."""
    print(f"ACTIVANDO AGENTE ESPECIALISTA VISUAL para: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            model_version = "lucataco/bakllava:452b2fa0b66d8acdf40e05a7f0af948f9c6065f6da5af22fce4cead99a26ff3d"
            output = replicate.run(
                model_version,
                input={"image": image_file, "question": "Describe this image in extreme detail. Analyze every component, label, and connection. Explain what it represents in a structured way."}
            )
            return output
    except Exception as e:
        print(f"Error en Agente Especialista Visual: {e}")
        return "No se pudo generar una descripción visual detallada."

def get_visual_response(visual_context: str, question: str, source_page: int) -> str:
    """Obtiene la respuesta final del LLM usando solo el contexto visual de alta calidad."""
    prompt = f"""
    Eres un experto analista de diagramas y arquitecturas. Tu única tarea es responder la pregunta del usuario basándote en la siguiente descripción de una imagen.

    DESCRIPCIÓN DE LA IMAGEN:
    {visual_context}

    PREGUNTA DEL USUARIO:
    {question}

    RESPUESTA DETALLADA (explica la imagen basándote en la descripción):
    """
    response = groq_llm.invoke(prompt).content
    return f"{response}\n\n[Fuente: Página {source_page}]"

def query_chain(chain, user_input: str):
    """Orquesta el RAG, mostrando imágenes específicas solo cuando es necesario."""
    try:
        # Para cualquier pregunta, primero obtenemos la respuesta de texto.
        logger.debug(f"Ejecutando RAG para: {user_input}")
        result = chain.invoke({"query": user_input})
        
        image_url = None
        source_documents = result.get("source_documents", [])
        
        # Lógica de visualización inteligente
        if source_documents and user_wants_image(user_input):
            top_doc = source_documents[0]
            metadata = top_doc.metadata
            source_filename, page_number = metadata.get("source"), metadata.get("page_number")

            if source_filename and page_number:
                # --- LÓGICA DE IMAGEN ESPECÍFICA ---
                # Buscamos la imagen más grande en la página relevante, asumiendo que es el diagrama principal.
                image_filename_to_show = f"{os.path.splitext(source_filename)[0]}_p{page_number}_full.png" # Por defecto, la página completa
                
                pdf_path = UPLOAD_DIR / source_filename
                if pdf_path.exists():
                    pdf_doc = fitz.open(pdf_path)
                    if page_number <= len(pdf_doc):
                        page = pdf_doc[page_number - 1]
                        images_on_page = page.get_images(full=True)
                        if images_on_page:
                            largest_image = max(images_on_page, key=lambda img: fitz.Pixmap(pdf_doc, img[0]).size)
                            img_index = images_on_page.index(largest_image)
                            image_filename_to_show = f"{os.path.splitext(source_filename)[0]}_p{page_number}_img{img_index}.png"

                relative_path = os.path.join("static/images", image_filename_to_show).replace("\\", "/")
                image_url = f"{BASE_URL}/{relative_path}"
                logger.debug(f"Mostrando imagen específica: {image_filename_to_show}")

        return {
            "response": result["result"],
            "sources": [doc.metadata.get("source", "N/A") for doc in source_documents],
            "image_url": image_url
        }
        
    except Exception as e:
        logger.exception("Error en query_chain")
        raise