# client/config.py
import streamlit as st

# When running locally, it will default to localhost.
# When deployed on Streamlit Cloud, it will read the API_URL from the configured secrets.
API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000")