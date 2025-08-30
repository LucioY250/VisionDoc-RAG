# En modules/query_handlers.py
from logger import logger

# Asegúrate de que esta línea esté al principio
BASE_URL = "http://127.0.0.1:8000"

def query_chain(chain, user_input: str):
    try:
        # ... (código para llamar a la cadena) ...
        result = chain({"query": user_input})
        
        image_url = None
        source_documents = result["source_documents"]

        if source_documents:
            top_doc = source_documents[0]
            if top_doc.metadata.get("type") == "image":
                image_path = top_doc.metadata.get("image_path")
                if image_path:
                    # ESTA ES LA LÓGICA CRÍTICA
                    relative_path = image_path.replace("\\", "/").lstrip("./")
                    image_url = f"{BASE_URL}/{relative_path}"

        response = {
            "response": result["result"],
            "sources": [doc.metadata.get("source", "") for doc in source_documents],
            "image_url": image_url
        }
        
        return response
    except Exception as e:
        logger.exception("Error in query_chain")
        raise