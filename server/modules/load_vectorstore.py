# En modules/load_vectorstore.py

import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
import fitz  # PyMuPDF
import replicate
from unstructured.partition.pdf import partition_pdf

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq

load_dotenv()

# --- CONFIGURACIÓN DE RUTAS ---
SERVER_ROOT = Path(__file__).parent.parent
PERSIST_DIR = SERVER_ROOT / "chroma_store"
UPLOAD_DIR = SERVER_ROOT / "uploaded_pdfs"
IMAGE_SAVE_DIR = SERVER_ROOT / "static" / "images"

# Aseguramos que los directorios existan al cargar el módulo
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)


# --- HERRAMIENTAS DE PROCESAMIENTO (SINGLETONS) ---
# Cargamos los modelos una sola vez para reutilizarlos.
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
groq_llm = ChatGroq(groq_api_key=os.environ.get("GROQ_API_KEY"), model_name="llama3-8b-8192", temperature=0)

# ==============================================================================
# FUNCIONES DE ENRIQUECIMIENTO (EL NÚCLEO MULTI-VECTOR)
# ==============================================================================

def summarize_text(text: str, filename: str, page_num: int) -> str:
    """Usa un LLM rápido para limpiar y resumir el texto de OCR en un párrafo coherente."""
    print(f"Resumiendo texto de {filename}, página {page_num}...")
    prompt = f"""
    Eres un experto en síntesis de información. El siguiente texto fue extraído de la página {page_num} del documento '{filename}' mediante OCR. 
    Tu tarea es limpiarlo, corregir errores evidentes y resumirlo en un párrafo conciso y denso en información que capture la esencia de la página.
    No añadas información que no esté presente.
    
    Texto extraído:
    ```{text}```
    
    Resumen coherente:
    """
    try:
        summary = groq_llm.invoke(prompt).content
        print(f"Resumen de texto generado para página {page_num}.")
        return summary
    except Exception as e:
        print(f"Error al resumir texto de página {page_num}: {e}")
        return text  # Si falla, devuelve el texto original

def describe_image(image_path: str, filename: str, page_num: int) -> str:
    """Usa un VLM (BakLLaVA en Replicate) para describir la estructura visual de una imagen."""
    print(f"Describiendo visualmente {filename}, página {page_num}...")
    try:
        with open(image_path, "rb") as image_file:
            model_version = "lucataco/bakllava:452b2fa0b66d8acdf40e05a7f0af948f9c6065f6da5af22fce4cead99a26ff3d"
            output = replicate.run(
                model_version,
                input={
                    "image": image_file,
                    "question": f"This is page {page_num} of the document '{filename}'. Describe the visual layout and structure. Identify key elements like diagrams, charts, tables, or important headings. Do not transcribe the text."
                }
            )
            print(f"Descripción visual generada para página {page_num}.")
            return output
    except Exception as e:
        print(f"Error al describir imagen de página {page_num}: {e}")
        return "No visual description could be generated."

# ==============================================================================
# FUNCIÓN DE INGESTIÓN PRINCIPAL (PIPELINE MULTI-VECTOR)
# ==============================================================================

def load_vectorstore(uploaded_files):
    file_paths = []
    for file in uploaded_files:
        save_path = UPLOAD_DIR / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    docs_for_vectorstore = []
    for path in file_paths:
        filename = os.path.basename(path)
        print(f"Iniciando procesamiento Multi-Vector para: {path}")

        # 1. Extracción de texto de alta calidad con Unstructured
        elements = partition_pdf(path, strategy="hi_res", infer_table_structure=True)
        
        # 2. Agrupación del contenido de texto por página
        pages_content = {}
        for el in elements:
            page_num = el.metadata.page_number
            if page_num not in pages_content:
                pages_content[page_num] = ""
            pages_content[page_num] += "\n\n" + str(el)

        # 3. Guardado de la imagen visual de cada página
        pdf_doc = fitz.open(path)
        for page_num, page in enumerate(pdf_doc):
            pix = page.get_pixmap(dpi=200)
            image_filename = f"{os.path.splitext(filename)[0]}_p{page_num + 1}_full.png"
            image_path = IMAGE_SAVE_DIR / image_filename
            pix.save(image_path)
        print(f"Guardadas {len(pdf_doc)} imágenes de página para {filename}.")

        # 4. Generación y almacenamiento de MÚLTIPLES representaciones por página
        for page_num, content in pages_content.items():
            image_path = IMAGE_SAVE_DIR / f"{os.path.splitext(filename)[0]}_p{page_num}_full.png"
            
            # --- Representación 1: Resumen del Texto (Anzuelo Conceptual) ---
            text_summary = summarize_text(content, filename, page_num)
            docs_for_vectorstore.append(Document(
                page_content=text_summary,
                metadata={"source": filename, "page_number": page_num, "content_type": "text_summary"}
            ))
            
            # --- Representación 2: Resumen Visual (Anzuelo Estructural) ---
            if image_path.exists():
                visual_summary = describe_image(str(image_path), filename, page_num)
                docs_for_vectorstore.append(Document(
                    page_content=visual_summary,
                    metadata={"source": filename, "page_number": page_num, "content_type": "visual_summary"}
                ))

    # 5. Creación del Vectorstore único con todas las representaciones
    # Se sobrescribe el vectorstore en cada subida para este reto.
    vectorstore = Chroma.from_documents(docs_for_vectorstore, embeddings, persist_directory=str(PERSIST_DIR))

    print("Proceso de ingestión con Multi-Vector Retriever completado.")
    # Ya no necesitamos el docstore, el retriever simple es suficiente ahora
    return vectorstore