import os
import streamlit as st
import numpy as np
import pandas as pd
import requests
from groq import Groq

# 1. Page Config
st.set_page_config(page_title="Global Diabetes Analytics", page_icon="🩺", layout="wide")

# 2. Tab Initialization (Make sure variable names match!)
tab_ml, tab_chat, tab_who, tab_daly, tab_overview = st.tabs([
    "🩺 AI Patient Predictor", 
    "💬 AI Clinical Assistant",
    "🌐 Live WHO Global Data", 
    "📊 State-wise Cost per DALY", 
    "🧠 ML Feature Importance"
])

# ---------------------------------------------------------
# TAB 1: AI PATIENT PREDICTOR
# ---------------------------------------------------------
with tab_ml:
    st.header("Patient Clinical Risk Calculator")
    # Tab 1 Code...

# ---------------------------------------------------------
# TAB 2: AI CLINICAL CHATBOT
# ---------------------------------------------------------
with tab_chat:
    st.header("AI Healthcare Assistant")
    st.write("Ask any questions regarding diabetes, diet, exercise, clinical metrics, or general health.")

    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        st.warning("⚠️ Please configure GROQ_API_KEY in Streamlit Secrets to enable the live AI model.")
    else:
        client = Groq(api_key=groq_api_key)

        system_instruction = {
            "role": "system",
            "content": (
                "You are an expert AI Clinical Health Assistant specializing in metabolic health, diabetes, and exercise science. "
                "Provide detailed, accurate, empathetic, and evidence-based answers to any patient query. "
                "Always clarify that your guidance is for informational purposes and does not replace official medical diagnosis."
            )
        }

        if "messages" not in st.session_state:
            st.session_state.messages = [
                system_instruction,
                {"role": "assistant", "content": "Hello! I am your AI Clinical Assistant. How can I assist you today?"}
            ]

        for msg in st.session_state.messages[1:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_input := st.chat_input("Type any medical or lifestyle query here..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing query..."):
                    try:
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=st.session_state.messages,
                            temperature=0.4,
                            max_tokens=800
                        )
                        reply = response.choices[0].message.content
                        st.write(reply)
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.error(f"Error communicating with AI model: {e}")
