# /components/history_download.py

import streamlit as st

def render_history_download():
    """
    Renders a download button for the chat history if messages exist.
    """
    # Check if the messages list exists and is not empty
    if "messages" in st.session_state and st.session_state.messages:
        
        # Format the chat history into a clean string
        chat_text = "\n\n".join(
            [f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages]
        )
        
        st.download_button(
            "Download Chat History",
            chat_text,
            file_name="chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )