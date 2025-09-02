# /components/upload.py

import streamlit as st
from utils.api import upload_pdfs_api

def render_uploader():
    """
    Renders the PDF uploader widget in the Streamlit sidebar.
    Handles the file upload logic and calls the backend API.
    """
    st.header("Upload PDFs")
    
    uploaded_files = st.file_uploader(
        "Upload one or more PDF documents",
        type="pdf",
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if st.button("Upload to DB", use_container_width=True) and uploaded_files:
        with st.spinner("Processing documents... This may take a few minutes."):
            response = upload_pdfs_api(uploaded_files)
            if response.status_code == 200:
                st.success("Documents processed successfully!")
                st.rerun() # Rerun to update the app state
            else:
                st.error(f"Error: {response.text}")