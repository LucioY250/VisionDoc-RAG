# En utils/api.py
import requests
from config import API_URL

def upload_pdfs_api(files):
    """Envía los archivos PDF al backend para su procesamiento."""
    files_payload = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
    return requests.post(f"{API_URL}/upload_pdfs/", files=files_payload)

def ask_question(question):
    """Llama al endpoint de consulta estándar."""
    return requests.post(f"{API_URL}/ask/", data={"question": question})