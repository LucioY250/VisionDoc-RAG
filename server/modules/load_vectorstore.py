# En modules/load_vectorstore.py
import os
from pathlib import Path
from dotenv import load_dotenv
import fitz
from unstructured.partition.pdf import partition_pdf
# --- LA CORRECCIÓN DEFINITIVA: Eliminamos los imports frágiles ---
from unstructured.cleaners.core import clean_extra_whitespace
# No importamos UnstructuredIOError, ya no es necesario
# --- FIN DE LA CORRECCIÓN ---

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
import replicate

load_dotenv()

SERVER_ROOT = Path(__file__).parent.parent
PERSIST_DIR = SERVER_ROOT / "chroma_store"
UPLOAD_DIR = SERVER_ROOT / "uploaded_pdfs"
IMAGE_SAVE_DIR = SERVER_ROOT / "static" / "images"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

def describe_image(image_path: str) -> str:
    print(f"Describing image element at: {image_path}")
    try:
        with open(image_path, "rb") as image_file:
            model_version = "lucataco/bakllava:452b2fa0b66d8acdf40e05a7f0af948f9c6065f6da5af22fce4cead99a26ff3d"
            question = "This is an image element from a document. Describe it in extreme detail. If it is an architecture diagram or flowchart, transcribe all text from every node and explain what each component does and how they are connected. Be structured and exhaustive."
            output = replicate.run(model_version, input={"image": image_file, "question": question})
            return output
    except Exception as e:
        print(f"Error describing image element: {e}")
        return "No visual description could be generated."

def load_vectorstore(uploaded_files):
    file_paths = [UPLOAD_DIR / f.filename for f in uploaded_files]
    for file, path in zip(uploaded_files, file_paths):
        with open(path, "wb") as f: f.write(file.file.read())

    atomic_docs = []
    for path in file_paths:
        filename = os.path.basename(path)
        print(f"Starting Atomic ingestion for: {path}")
        
        try:
            elements = partition_pdf(
                str(path), 
                strategy="hi_res", 
                infer_table_structure=True, 
                extract_images_in_pdf=True,
                chunking_strategy="by_title"
            )
        # --- LA CORRECCIÓN DEFINITIVA: Capturamos cualquier excepción ---
        except Exception as e:
            print(f"Could not process PDF {filename} with Unstructured. Skipping. Error: {e}")
            continue
        # --- FIN DE LA CORRECCIÓN ---

        for el in elements:
            metadata = {"source": filename, "page_number": el.metadata.page_number}
            if el.category == "Image":
                image_id = el.id
                img_path = IMAGE_SAVE_DIR / f"{os.path.splitext(filename)[0]}_p{el.metadata.page_number}_{image_id}.jpg"
                try:
                    el.image.save(img_path, "JPEG")
                    description = describe_image(str(img_path))
                    atomic_docs.append(Document(page_content=description, metadata=metadata))
                except Exception as img_e:
                    print(f"Could not save or describe image {image_id}: {img_e}")
            else:
                text = clean_extra_whitespace(el.text)
                if len(text) > 30:
                    atomic_docs.append(Document(page_content=text, metadata=metadata))

    vectorstore = Chroma.from_documents(atomic_docs, embeddings, persist_directory=str(PERSIST_DIR))
    print("Atomic Ingestion complete.")
    return vectorstore