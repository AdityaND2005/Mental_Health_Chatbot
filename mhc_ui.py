"""
Streamlit UI
Mental Health RAG Chatbot
No terminal mode. No API mode. Pure UI.
"""

import os
import streamlit as st
import config
from rag_system import create_rag_system
from crisis_resources import format_crisis_resources


# ----------------------------------
# Load RAG once (important)
# ----------------------------------
@st.cache_resource
def load_rag():
    lightweight = os.getenv("LIGHTWEIGHT_CLASSIFIER") == "1"
    return create_rag_system(lightweight=lightweight)

def run_mhc_ui():
    
    rag = load_rag()


    # ----------------------------------
    # Page Config
    # ----------------------------------
    st.set_page_config(
        page_title="Mental Health Support Chatbot",
        page_icon="💚",
        layout="wide"
    )

    st.title("💚 Mental Health Support Chatbot")

    st.markdown("""
    A compassionate AI counselor providing supportive conversations.

    ⚠️ **This chatbot is not a substitute for professional help.**
    If you are in immediate danger, call emergency services.
    """)


    # ----------------------------------
    # Crisis Helplines Accordion
    # ----------------------------------
    with st.expander("🆘 Emergency Crisis Helplines (India)"):
        st.markdown(format_crisis_resources())


    # ----------------------------------
    # Session State (Chat Memory)
    # ----------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []


    # ----------------------------------
    # Display Chat History
    # ----------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


    # ----------------------------------
    # User Input
    # ----------------------------------
    if prompt := st.chat_input("Type your message here..."):

        # Show user message
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        try:
            response_data = rag.chat(prompt)

            bot_response = response_data["response"]

            if response_data.get("is_crisis", False):
                bot_response = (
                    "🚨 **CRISIS DETECTED - PLEASE SEE RESOURCES BELOW** 🚨\n\n"
                    + bot_response
                )

            st.session_state.messages.append(
                {"role": "assistant", "content": bot_response}
            )

            with st.chat_message("assistant"):
                st.markdown(bot_response)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


    # ----------------------------------
    # Sidebar Controls
    # ----------------------------------
    with st.sidebar:
        st.header("Options")

        if st.button("Reset Conversation"):
            rag.reset_conversation()
            st.session_state.messages = []
            st.rerun()