# modules/load_vectorstore.py

import os
import time
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

# --- Module-level Configuration ---
SERVER_ROOT = Path(__file__).parent.parent
PERSIST_DIR = SERVER_ROOT / "chroma_store"
UPLOAD_DIR = SERVER_ROOT / "uploaded_pdfs"
IMAGE_SAVE_DIR = SERVER_ROOT / "static" / "images"

# Ensure necessary directories exist on module load
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# --- Singleton AI Models (loaded once for efficiency) ---
# State-of-the-art multilingual embedding model for maximum retrieval precision.
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
# Powerful LLM for high-quality text summarization during ingestion.
groq_llm = ChatGroq(groq_api_key=os.environ.get("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile", temperature=0)


def summarize_text(content: str, filename: str, page_num: int) -> str:
    """
    Uses a powerful LLM to clean and summarize raw OCR text into a coherent paragraph.
    This creates a high-quality textual representation for each page.
    """
    print(f"Summarizing text for {filename}, page {page_num}...")
    prompt = f"Summarize the following OCR text from page {page_num} of '{filename}' into a concise, information-dense paragraph. Correct obvious OCR errors. Text: ```{content}```"
    try:
        return groq_llm.invoke(prompt).content
    except Exception as e:
        print(f"Error summarizing page {page_num}: {e}")
        return content  # Return raw text as a fallback

def describe_image(image_path: str, filename: str, page_num: int) -> str:
    """
    Uses a VLM via Replicate API to generate a detailed visual description of a page image.
    Includes a retry mechanism to handle API timeouts and instability.
    """
    print(f"Describing visuals for {filename}, page {page_num}...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(image_path, "rb") as image_file:
                model_version = "lucataco/bakllava:452b2fa0b66d8acdf40e05a7f0af948f9c6065f6da5af22fce4cead99a26ff3d"
                question = f"This is page {page_num} of the document '{filename}'. Describe it in extreme detail. If it is a technical diagram or architecture flowchart, you MUST transcribe all text from every node and explain what each component does and how they are connected. Be structured and exhaustive."
                output = replicate.run(model_version, input={"image": image_file, "question": question})
                return output
        except Exception as e:
            if "timed out" in str(e).lower() and attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed for page {page_num}: Timeout. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Error describing page {page_num} after {max_retries} attempts: {e}")
                return "No visual description could be generated."

def process_page_hybrid(page_data: tuple) -> Document:
    """
    Processes a single page by generating both textual and visual summaries in parallel.
    Fuses them into a single rich context for the vector store.
    """
    content, filename, page_num, image_path = page_data
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_text = executor.submit(summarize_text, content, filename, page_num)
        future_visual = executor.submit(describe_image, str(image_path), filename, page_num)
        text_summary = future_text.result()
        visual_summary = future_visual.result()

    fused_content = f"[TEXTUAL SUMMARY OF PAGE {page_num}]:\n{text_summary}\n\n[VISUAL DESCRIPTION OF PAGE {page_num}]:\n{visual_summary}"
    return Document(page_content=fused_content, metadata={"source": filename, "page_number": page_num})

def load_vectorstore(uploaded_files: list):
    """
    Main ingestion pipeline. Processes PDFs using a hybrid, parallelized approach for maximum quality and optimized speed.
    """
    file_paths = [UPLOAD_DIR / f.filename for f in uploaded_files]
    for file, path in zip(uploaded_files, file_paths):
        with open(path, "wb") as f: f.write(file.file.read())

    hybrid_docs = []
    for path in file_paths:
        filename = os.path.basename(path)
        print(f"Starting Hybrid & Parallel ingestion for: {path}")

        elements = partition_pdf(str(path), strategy="hi_res", infer_table_structure=True)
        pages_content = {}
        for el in elements:
            page_num = el.metadata.page_number
            if page_num not in pages_content:
                pages_content[page_num] = ""
            pages_content[page_num] += "\n\n" + str(el)

        tasks_to_run_in_parallel = []
        pdf_doc = fitz.open(path)
        for page in pdf_doc:
            page_num = page.number + 1
            image_path_full = IMAGE_SAVE_DIR / f"{os.path.splitext(filename)[0]}_p{page_num}_full.png"
            page.get_pixmap(dpi=200).save(image_path_full)

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                if xref:
                    img_pix = fitz.Pixmap(pdf_doc, xref)
                    if img_pix.n - img_pix.alpha >= 3:
                        img_path_specific = IMAGE_SAVE_DIR / f"{os.path.splitext(filename)[0]}_p{page_num}_img{img_index}.png"
                        img_pix.save(img_path_specific)

            if page_num in pages_content:
                tasks_to_run_in_parallel.append((pages_content[page_num], filename, page_num, image_path_full))
        
        # Parallelize the AI-heavy processing across all pages
        with ThreadPoolExecutor(max_workers=22) as executor:
            future_to_page = {executor.submit(process_page_hybrid, task): task for task in tasks_to_run_in_parallel}
            for future in as_completed(future_to_page):
                try:
                    hybrid_docs.append(future.result())
                except Exception as exc:
                    print(f"A processing task generated an error: {exc}")

    vectorstore = Chroma.from_documents(hybrid_docs, embeddings, persist_directory=str(PERSIST_DIR))
    print("High-Definition Hybrid & Parallel ingestion complete.")
    return vectorstore