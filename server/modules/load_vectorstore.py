# En modules/load_vectorstore.py

# --- IMPORTS NECESARIOS ---
import os
from pathlib import Path
from dotenv import load_dotenv
import fitz  # PyMuPDF
import replicate # Volvemos a usar Replicate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

load_dotenv()

# --- CONFIGURACIÓN ---
PERSIST_DIR = "./chroma_store"
UPLOAD_DIR = "./uploaded_pdfs"
IMAGE_SAVE_DIR = "./static/images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# ==============================================================================
# FUNCIÓN DE DESCRIPCIÓN DE IMÁGENES CON REPLICATE (ACTUALIZADA Y FIABLE)
# ==============================================================================
def describe_image_with_replicate(image_path: str) -> str:
    print(f"Describiendo imagen con Replicate (Modelo: BakLLaVA): {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            # Usamos el identificador exacto y verificado que encontraste
            model_version = "lucataco/bakllava:452b2fa0b66d8acdf40e05a7f0af948f9c6065f6da5af22fce4cead99a26ff3d"
            
            output = replicate.run(
                model_version,
                input={
                    "image": image_file,
                    "question": "Describe this image in detail. If it is a diagram or a chart, explain its components, labels, and the relationships between them."
                    # Nota: Este modelo usa el parámetro "question" en lugar de "prompt"
                }
            )
            # BakLLaVA devuelve una única cadena, no un generador, así que esto es más directo
            description = output
            
            print(f"Descripción generada: {description[:150]}...")
            return description
    except Exception as e:
        if "402" in str(e):
             print("\nERROR CRÍTICO: Error 402 - Crédito Insuficiente. Revisa tu cuenta de Replicate.")
        else:
             print(f"Error describiendo la imagen {image_path} con Replicate: {e}")
        return "No description could be generated for this image."
# ==============================================================================
# FUNCIÓN DE INGESTIÓN PRINCIPAL
# ==============================================================================
def load_vectorstore(uploaded_files):
    file_paths = []
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR) / file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))

    all_docs_to_process = []
    for path in file_paths:
        pdf_doc = fitz.open(path)
        filename = os.path.basename(path)

        # Extraer TEXTO
        for page_num, page in enumerate(pdf_doc):
            text = page.get_text()
            if text:
                metadata = {"source": filename, "page": page_num + 1, "type": "text"}
                all_docs_to_process.append(Document(page_content=text, metadata=metadata))

        # Extraer IMÁGENES
        for page_num, page in enumerate(pdf_doc):
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf_doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                image_filename = f"{os.path.splitext(filename)[0]}_p{page_num+1}_img{img_index}.png"
                image_path = os.path.join(IMAGE_SAVE_DIR, image_filename)
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                
                # Usamos nuestra función de Replicate
                description = describe_image_with_replicate(image_path)
                
                metadata = {
                    "source": filename, 
                    "page": page_num + 1, 
                    "type": "image",
                    "image_path": image_path
                }
                all_docs_to_process.append(Document(page_content=description, metadata=metadata))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(all_docs_to_process)

    embeddings = HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L12-v2")

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        vectorstore.add_documents(texts)
    else:
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )

    print("Proceso de ingestión completado.")
    return vectorstore