# En modules/load_vectorstore.py

import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
import fitz  # PyMuPDF
import replicate
from unstructured.partition.pdf import partition_pdf
from concurrent.futures import ThreadPoolExecutor, as_completed

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
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# --- HERRAMIENTAS DE PROCESAMIENTO (MÁXIMA CALIDAD) ---
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
groq_llm = ChatGroq(groq_api_key=os.environ.get("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile", temperature=0)

# ==============================================================================
# FUNCIONES DE ENRIQUECIMIENTO
# ==============================================================================
def summarize_text(content, filename, page_num):
    print(f"Resumiendo texto de {filename}, pág {page_num}...")
    prompt = f"Resume el siguiente texto extraído de la página {page_num} del documento '{filename}' en un párrafo conciso y denso en información. Texto: ```{content}```"
    try:
        return groq_llm.invoke(prompt).content
    except Exception as e:
        print(f"Error al resumir pág {page_num}: {e}")
        return content

def describe_image(image_path, filename, page_num):
    print(f"Describiendo visualmente {filename}, pág {page_num}...")
    try:
        with open(image_path, "rb") as image_file:
            model_version = "lucataco/bakllava:452b2fa0b66d8acdf40e05a7f0af948f9c6065f6da5af22fce4cead99a26ff3d"
            output = replicate.run(model_version, input={"image": image_file, "question": f"This is page {page_num} of the document '{filename}'. Describe the visual layout, structure, and key elements like diagrams or charts. Do not transcribe text."})
            return output
    except Exception as e:
        print(f"Error al describir pág {page_num}: {e}")
        return "No visual description could be generated."

def process_page(page_data):
    """Función que procesa una única página para ser ejecutada en paralelo."""
    content, filename, page_num, image_path = page_data
    
    text_summary = summarize_text(content, filename, page_num)
    visual_summary = describe_image(str(image_path), filename, page_num)

    fused_content = f"[RESUMEN TEXTUAL DE LA PÁGINA {page_num}]:\n{text_summary}\n\n[DESCRIPCIÓN VISUAL DE LA PÁGINA {page_num}]:\n{visual_summary}"
    
    return Document(
        page_content=fused_content,
        metadata={"source": filename, "page_number": page_num}
    )

# ==============================================================================
# FUNCIÓN DE INGESTIÓN PRINCIPAL (HÍBRIDA Y PARALELA)
# ==============================================================================
def load_vectorstore(uploaded_files):
    file_paths = [str(UPLOAD_DIR / f.filename) for f in uploaded_files]
    for file, path in zip(uploaded_files, file_paths):
        with open(path, "wb") as f: f.write(file.file.read())

    hybrid_docs = []
    for path in file_paths:
        filename = os.path.basename(path)
        print(f"Iniciando procesamiento Híbrido y Paralelo para: {path}")

        elements = partition_pdf(path, strategy="hi_res", infer_table_structure=True)
        pages_content = {}
        for el in elements:
            page_num = el.metadata.page_number
            if page_num not in pages_content: pages_content[page_num] = ""
            pages_content[page_num] += "\n\n" + str(el)
        
        tasks = []
        pdf_doc = fitz.open(path)
        for page_num_fitz, page in enumerate(pdf_doc):
            page_num = page_num_fitz + 1
            pix = page.get_pixmap(dpi=200)
            image_filename = f"{os.path.splitext(filename)[0]}_p{page_num}_full.png"
            image_path = IMAGE_SAVE_DIR / image_filename
            pix.save(image_path)
            
            # Extraemos y guardamos imágenes específicas (diagramas)
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                if xref:
                    img_pix = fitz.Pixmap(pdf_doc, xref)
                    if img_pix.n - img_pix.alpha >= 3: # Solo imágenes a color, no máscaras
                         img_filename_specific = f"{os.path.splitext(filename)[0]}_p{page_num}_img{img_index}.png"
                         img_path_specific = IMAGE_SAVE_DIR / img_filename_specific
                         img_pix.save(img_path_specific)

            if page_num in pages_content:
                tasks.append((pages_content[page_num], filename, page_num, image_path))
        
        # --- LA MAGIA DE LA PARALELIZACIÓN ---
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_page = {executor.submit(process_page, task): task for task in tasks}
            for future in as_completed(future_to_page):
                try:
                    hybrid_docs.append(future.result())
                except Exception as exc:
                    print(f"Una tarea de procesamiento generó un error: {exc}")

    vectorstore = Chroma.from_documents(hybrid_docs, embeddings, persist_directory=str(PERSIST_DIR))
    print("Proceso de ingestión Híbrido y Paralelo completado.")
    return vectorstore