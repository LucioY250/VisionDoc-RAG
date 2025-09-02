# VisionDoc RAG: Advanced Multimodal RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.2-orange.svg)
![Docker](https://img.shields.io/badge/Docker-24.0-blue.svg)

An enterprise-grade, multimodal RAG (Retrieval-Augmented Generation) chatbot designed to process and answer questions about complex documents containing both text and images, including scanned PDFs. This project was developed as a solution to a comprehensive technical challenge, prioritizing precision, speed, and a high-quality user experience.

## ✨ Key Features

This isn't just a standard RAG pipeline. It's a robust system built with advanced techniques to overcome common failures in document processing:

*   **🧠 High-Definition Multimodal Ingestion:** Instead of relying on basic text extraction, the system creates a rich, "fused context" for each page by:
    1.  Performing high-quality OCR on scanned documents using `unstructured.io`.
    2.  Generating an intelligent **textual summary** of each page's content using a powerful LLM (Groq Llama 3.1 70B).
    3.  Generating a detailed **visual description** of each page's layout and diagrams using a VLM (BakLLaVA on Replicate).
*   **🚀 Optimized Ingestion Speed:** Despite the heavy AI processing, the ingestion pipeline is highly parallelized using a `ThreadPoolExecutor`, processing all pages and API calls concurrently to reduce the total time from over 10 minutes to **~3-4 minutes**.
*   **🎯 State-of-the-Art Retrieval:**
    *   **Multilingual Embeddings:** Utilizes the powerful `BAAI/bge-m3` model, ensuring top-tier semantic understanding across multiple languages.
    *   **Two-Phase Retrieval:** Employs a `RerankingRetriever` with `BAAI/bge-reranker-v2-m3` to first fetch a broad set of candidates and then re-rank them for maximum relevance, guaranteeing the most accurate context is sent to the LLM.
*   **🗣️ Bilingual & Language-Aware:** The system automatically detects the user's query language, translates it to English for optimal retrieval accuracy against the English documents, and then instructs the final LLM to respond in the user's original language.
*   **🖼️ Precise Multimodality:** The chatbot doesn't just show the whole page for a visual query. It intelligently extracts and displays the **specific sub-image** (like a diagram) most relevant to the question, providing a clean and focused user experience.
*   **⚡ Modern & Scalable Tech Stack:** Built with a FastAPI backend for robustness and a Streamlit frontend for rapid UI development, all containerized with Docker for easy deployment.

## 🛠️ Tech Stack

| Component             | Technology / Service                                  | Purpose                                           |
| --------------------- | ----------------------------------------------------- | ------------------------------------------------- |
| **Backend**           | FastAPI, Uvicorn                                      | Robust, high-performance asynchronous API server. |
| **Frontend**          | Streamlit                                             | Interactive and fast UI development.              |
| **Core AI / RAG**     | LangChain                                             | Orchestration of the RAG pipeline.                |
| **Document Parsing**  | `unstructured.io`, `PyMuPDF`                          | High-quality OCR and PDF element extraction.      |
| **Embeddings**        | `BAAI/bge-m3`                                         | State-of-the-art multilingual embeddings.         |
| **Re-ranking**        | `BAAI/bge-reranker-v2-m3`                             | Precision enhancement for retrieval.              |
| **Text Summarization**| Groq API (`llama-3.3-70b-versatile`)                  | High-quality text summarization during ingestion. |
| **Visual Description**| Replicate API (`lucataco/bakllava`)                   | Detailed diagram and image description.           |
| **Vector Database**   | ChromaDB                                              | Local, persistent vector storage.                 |
| **Containerization**  | Docker                                                | Packaging the application for deployment.         |
| **Deployment Target** | Render (Backend) & Streamlit Community Cloud (Frontend) | Cloud hosting with persistent storage.            |

## 🏗️ Architecture Overview

The system is designed around two distinct pipelines: a one-time, high-quality Ingestion Pipeline and a real-time, low-latency Querying Pipeline.

### 1. Ingestion Pipeline (Per Document)

1.  **PDF Parsing:** The document is loaded, and `unstructured.io` performs high-resolution OCR to extract all text elements. `PyMuPDF` extracts visual page images and specific sub-images (like diagrams).
2.  **Parallel Enrichment:** For each page, two AI tasks are executed concurrently:
    *   **Text Summarization:** The raw OCR text is sent to Groq's Llama 3.1 70B model to be cleaned and summarized.
    *   **Visual Description:** The page image is sent to the BakLLaVA model on Replicate for a detailed visual analysis.
3.  **Context Fusion:** The textual summary and visual description are fused into a single, rich text block.
4.  **Embedding:** The fused context is converted into a high-definition vector using the `bge-m3` model.
5.  **Storage:** The embedding and its associated metadata (source file, page number) are stored in a persistent ChromaDB vector database.

### 2. Querying Pipeline (Per Question)

1.  **Pre-processing:** The user's query is analyzed. If it's not in English, it's translated using a fast LLM (`llama-3.1-8b-instant`).
2.  **Retrieval:** The translated query is embedded with `bge-m3` and used to find the top 10 most relevant document pages from ChromaDB.
3.  **Re-ranking:** The `bge-reranker-v2-m3` model re-evaluates these 10 candidates against the query and selects the top 3 most precise results.
4.  **Generation:** The fused context from these top 3 pages is passed to the powerful `llama-3.1-70b-versatile` model with a detailed prompt, which generates the final answer.
5.  **Multimodal Logic:** If the user's query expressed visual intent, the system identifies the most relevant sub-image from the retrieved page and includes its URL in the final response for the frontend to display.

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

*   Python 3.10+
*   Git
*   **System Dependencies:** This project relies on `unstructured.io`, which requires the following to be installed on your system:
    *   **Poppler:** For PDF rendering.
    *   **Tesseract:** For OCR.
    *   Ensure they are correctly installed and available in your system's PATH.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/VisionDoc-RAG.git
cd VisionDoc-RAG
```

### 2. Set Up the Environment

#### Navigate to the server directory
```bash
cd server
```
#### Create and activate the virtual environment´

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a .env file in the server directory. Copy the contents of .env.example and fill in your API keys.

server/.env.example:

```bash
GROQ_API_KEY="gsk_..."
REPLICATE_API_TOKEN="r8_..."
```

### 5. Run the Application
You'll need two separate terminals, both with the virtual environment activated.

Terminal 1: Start the Backend Server
```bash
# From the server/ directory
uvicorn main:app --reload
Terminal 2: Start the Frontend App
```
```bash
# From the root project directory (VisionDoc-RAG/)
# Assuming your frontend files are in a 'client/' directory
streamlit run client/app.py
```

### 6. Usage
- Open your browser to the Streamlit URL (usually http://localhost:8501).
- Use the sidebar to upload one or more PDF documents.
- Click the "Upload to DB" button and wait for the ingestion process to complete (monitor the backend terminal for progress).
- Once ingestion is complete, start asking questions in the chat interface!