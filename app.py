# app.py (FINAL UNIFIED ENTRY)

import streamlit as st
from mhc_ui import run_mhc_ui
from disease_ui import run_disease_ui

st.set_page_config(
    page_title="Health AI Assistant",
    page_icon="🧠",
    layout="wide"
)

st.sidebar.title("Navigation")

mode = st.sidebar.radio(
    "Choose Module:",
    ["Mental Health Chatbot", "Disease Prediction"]
)

if mode == "Mental Health Chatbot":
    run_mhc_ui()

elif mode == "Disease Prediction":
    run_disease_ui()