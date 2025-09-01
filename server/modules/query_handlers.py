# En modules/query_handlers.py
import os
from pathlib import Path
from logger import logger
import fitz
from modules.load_vectorstore import UPLOAD_DIR

SERVER_ROOT = Path(__file__).parent.parent
BASE_URL = "http://127.0.0.1:8000"

def user_wants_image(user_input: str) -> bool:
    keywords = ["show me", "diagram", "image", "graph", "architecture", "flow", "what does it look like", "muéstrame", "enséñame", "diagrama", "imagen", "gráfico", "arquitectura", "flujo"]
    return any(keyword in user_input.lower() for keyword in keywords)

def query_chain(chain, user_input: str):
    """
    Orquesta el RAG de una sola etapa, mostrando la imagen específica más relevante solo cuando es necesario.
    """
    try:
        logger.debug(f"Executing High-Quality RAG for: {user_input}")
        result = chain.invoke({"query": user_input})
        
        image_url = None
        source_documents = result.get("source_documents", [])
        
        if source_documents and user_wants_image(user_input):
            top_doc = source_documents[0]
            metadata = top_doc.metadata
            source_filename, page_number = metadata.get("source"), metadata.get("page_number")

            if source_filename and page_number:
                image_filename_to_show = f"{os.path.splitext(source_filename)[0]}_p{page_number}_full.png"
                pdf_path = UPLOAD_DIR / source_filename
                if pdf_path.exists():
                    pdf_doc = fitz.open(pdf_path)
                    if page_number <= len(pdf_doc):
                        page = pdf_doc[page_number - 1]
                        images_on_page = page.get_images(full=True)
                        if images_on_page:
                            largest_image_info = max(images_on_page, key=lambda img: fitz.Pixmap(pdf_doc, img[0]).size)
                            img_index = images_on_page.index(largest_image_info)
                            image_filename_to_show = f"{os.path.splitext(source_filename)[0]}_p{page_number}_img{img_index}.png"

                relative_path = os.path.join("static/images", image_filename_to_show).replace("\\", "/")
                image_url = f"{BASE_URL}/{relative_path}"
                logger.debug(f"Displaying specific image: {image_filename_to_show}")

        return {"response": result["result"], "sources": [doc.metadata.get("source", "N/A") for doc in source_documents], "image_url": image_url}
    except Exception as e:
        logger.exception("Error in query_chain")
        raise