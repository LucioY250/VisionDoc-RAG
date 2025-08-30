# En chatUI.py

import streamlit as st
from utils.api import ask_question

def render_chat():
    st.subheader("💬 Chat with your documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # -------------------------------------------------------------------------
    # NUEVO 1: MODIFICACIÓN DEL RENDERIZADO DEL HISTORIAL
    # Ahora el historial puede contener imágenes, así que lo renderizamos aquí.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Si el mensaje del asistente tiene una URL de imagen, la mostramos
            if "image_url" in msg and msg["image_url"]:
                st.image(msg["image_url"])
    # -------------------------------------------------------------------------

    # Input del usuario
    user_input = st.chat_input("Type your question here...")
    if user_input:
        # Añadir mensaje del usuario a la UI y al historial
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Llamar al backend
        with st.spinner("Thinking..."):
            response = ask_question(user_input)
        
        if response.status_code == 200:
            data = response.json()
            answer = data["response"]
            
            # -----------------------------------------------------------------
            # NUEVO 2: EXTRACCIÓN DE LA URL DE LA IMAGEN
            # Buscamos el campo 'image_url' en la respuesta del backend.
            image_url = data.get("image_url", None)
            # -----------------------------------------------------------------

            # Renderizar la respuesta del asistente en la UI
            with st.chat_message("assistant"):
                st.markdown(answer)
                
                # -------------------------------------------------------------
                # NUEVO 3: RENDERIZADO CONDICIONAL DE LA IMAGEN
                # Si hemos recibido una URL, mostramos la imagen.
                if image_url:
                    st.image(image_url)
                # -------------------------------------------------------------
            
            sources = data.get("sources", [])
            if sources:
                st.markdown("📄 **Sources:**")
                for src in sources:
                    st.markdown(f"- `{src}`")
            
            # -----------------------------------------------------------------
            # NUEVO 4: GUARDADO MULTIMODAL EN EL HISTORIAL
            # Guardamos tanto el texto como la URL de la imagen en el estado de la sesión.
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "image_url": image_url # Guardamos la URL
            })
            # -----------------------------------------------------------------
        else:
            st.error(f"Error: {response.text}")