# En modules/load_vectorstore.py

import os
from pathlib import Path
from dotenv import load_dotenv
import fitz  # PyMuPDF
import uuid
from unstructured.partition.pdf import partition_pdf
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.storage import InMemoryStore

load_dotenv()

# --- MEJORA: Manejo de rutas robusto y centralizado ---
SERVER_ROOT = Path(__file__).parent.parent
PERSIST_DIR = SERVER_ROOT / "chroma_store"
UPLOAD_DIR = SERVER_ROOT / "uploaded_pdfs"
IMAGE_SAVE_DIR = SERVER_ROOT / "static" / "images"

# Aseguramos que los directorios existan al cargar el módulo
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

def load_vectorstore(uploaded_files):
    file_paths = []
    for file in uploaded_files:
        save_path = UPLOAD_DIR / file.filename
        with open(save_path, "wb") as f: f.write(file.file.read())
        file_paths.append(str(save_path))

    all_pages = []
    parent_ids = [] # << NUEVA LISTA para guardar los IDs de los padres
    for path in file_paths:
        filename = os.path.basename(path)
        print(f"Procesando documento con Unstructured: {path}")
        elements = partition_pdf(path, strategy="hi_res", infer_table_structure=True)

        pages_content = {}
        for el in elements:
            page_num = el.metadata.page_number
            if page_num not in pages_content: pages_content[page_num] = ""
            pages_content[page_num] += "\n\n" + str(el)

        for page_num, content in pages_content.items():
            parent_id = str(uuid.uuid4()) # << Generamos un ID único para cada página
            parent_ids.append(parent_id)
            metadata = {"source": filename, "page_number": page_num, "doc_id": parent_id}
            all_pages.append(Document(page_content=content, metadata=metadata))

        # Guardar imágenes (sin cambios)
        pdf_doc = fitz.open(path)
        for page_num, page in enumerate(pdf_doc):
            pix = page.get_pixmap(dpi=200)
            image_filename = f"{os.path.splitext(filename)[0]}_p{page_num + 1}_full.png"
            image_path = IMAGE_SAVE_DIR / image_filename
            pix.save(image_path)

    # El text splitter ahora crea los "chunks hijos"
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    # --- LA CORRECCIÓN CRÍTICA ---
    # Dividimos los documentos y asignamos los IDs de los padres a los hijos
    sub_docs = []
    for i, doc in enumerate(all_pages):
        _id = parent_ids[i]
        _sub_docs = child_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata["doc_id"] = _id
        sub_docs.extend(_sub_docs)

    embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    vectorstore = Chroma.from_documents(sub_docs, embeddings, persist_directory=str(PERSIST_DIR))
    
    docstore = InMemoryStore()
    docstore.mset(list(zip(parent_ids, all_pages))) # <<< Usamos los IDs para el docstore

    print("Proceso de ingestión con Parent-Document (con IDs) completado.")
    return vectorstore, docstore