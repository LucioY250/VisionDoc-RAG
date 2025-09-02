# /components/chatUI.py

import streamlit as st
from utils.api import ask_question
import requests

def render_chat():
    """
    Renders the main chat interface, including history and user input handling.
    """
    st.header("💬 Chat with your documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("image_url"):
                st.image(msg["image_url"])
            if msg.get("sources"):
                # Deduplicate sources and display them cleanly
                sources_text = ", ".join(list(set(msg["sources"])))
                st.caption(f"Sources: {sources_text}")

    # Handle new user input
    if user_input := st.chat_input("Type your question here..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        # Process and display the assistant's response
        with st.chat_message("assistant"):
            with st.spinner("Thinking... This may take a moment for complex questions."):
                response = ask_question(user_input)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get("response", "Sorry, I couldn't find an answer.")
                image_url = data.get("image_url")
                sources = data.get("sources", [])
                
                st.markdown(answer)
                if image_url:
                    st.image(image_url)
                if sources:
                    sources_text = ", ".join(list(set(sources)))
                    st.caption(f"Sources: {sources_text}")

                # Save the complete response to the session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "image_url": image_url,
                    "sources": sources
                })
            else:
                st.error(f"Error from backend: {response.text}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Sorry, there was an error processing your request."
                })