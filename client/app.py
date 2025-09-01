# En app.py
import streamlit as st
from components.upload import render_uploader
from components.chatUI import render_chat

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="VisionDoc RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INYECCIÓN DE CSS PARA UN LOOK MÁS PULIDO ---
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #1a1a2e;
    }
    .stChatMessage {
        background-color: #2a2a3e;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- LAYOUT DE LA APLICACIÓN ---
with st.sidebar:
    st.title("📄 VisionDoc RAG")
    st.caption("Your intelligent document assistant")
    render_uploader()
    # El botón de descarga del historial podría ir aquí también
    # from components.history_download import render_history_download
    # render_history_download()

st.container()
render_chat()