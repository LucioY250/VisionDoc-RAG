# /utils/api.py

import requests
from typing import List
import streamlit as st
from config import API_URL

def upload_pdfs_api(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> requests.Response:
    """
    Sends a list of uploaded PDF files to the backend's /upload_pdfs/ endpoint.

    Args:
        files: A list of Streamlit UploadedFile objects.

    Returns:
        The Response object from the requests.post call.
    """
    files_payload = [
        ("files", (file.name, file.getvalue(), "application/pdf")) for file in files
    ]
    return requests.post(f"{API_URL}/upload_pdfs/", files=files_payload)

def ask_question(question: str) -> requests.Response:
    """
    Sends a user's question to the backend's /ask/ endpoint.

    Args:
        question: The user's question as a string.

    Returns:
        The Response object from the requests.post call.
    """
    return requests.post(f"{API_URL}/ask/", data={"question": question})