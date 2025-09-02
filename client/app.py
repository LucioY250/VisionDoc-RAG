# /app.py

import streamlit as st
from components.upload import render_uploader
from components.chatUI import render_chat
from components.history_download import render_history_download

# --- Page Configuration ---
# Sets the browser tab title, icon, and layout for the application.
st.set_page_config(
    page_title="VisionDoc RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Injection ---
# Applies custom styles for a more polished and modern look.
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
        border-right: 1px solid #2a2a3a;
    }
    .stChatMessage {
        background-color: #2a2a3e;
        border-radius: 0.5rem;
        border: 1px solid #3a3a4e;
    }
</style>
""", unsafe_allow_html=True)


# --- Application Layout ---

# Sidebar for controls (file upload, etc.)
with st.sidebar:
    st.title("📄 VisionDoc RAG")
    st.caption("Your intelligent document assistant")
    
    render_uploader()
    
    st.divider()
    
    render_history_download()

# Main container for the chat interface
render_chat()