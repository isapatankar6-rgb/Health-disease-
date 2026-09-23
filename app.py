import os
from groq import Groq

# ---------------------------------------------------------
# TAB 2: TRUE MULTI-TURN AI CLINICAL CHATBOT
# ---------------------------------------------------------
with tab_chat:
    st.header("AI Healthcare Assistant")
    st.write("Ask any questions regarding diabetes, diet, exercise, clinical metrics, or general health.")

    # Initialize Groq client using Streamlit secrets or local environment key
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        st.warning("⚠️ Please configure `GROQ_API_KEY` in Streamlit Secrets to enable the live AI model.")
    else:
        client = Groq(api_key=groq_api_key)

        # System prompt setting clinical context and behavioral guidelines
        system_instruction = {
            "role": "system",
            "content": (
                "You are an expert AI Clinical Health Assistant specializing in metabolic health, diabetes, and exercise science. "
                "Provide detailed, accurate, empathetic, and evidence-based answers to any patient query. "
                "Always clarify that your guidance is for informational purposes and does not replace official medical diagnosis."
            )
        }

        # Initialize full message history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                system_instruction,
                {"role": "assistant", "content": "Hello! I am your AI Clinical Assistant. How can I assist you with your health or diabetes questions today?"}
            ]

        # Render conversation history (excluding the hidden system instruction)
        for msg in st.session_state.messages[1:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Handle new user input
        if user_input := st.chat_input("Type any medical or lifestyle query here..."):
            # Append user message to state
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            # Generate dynamic LLM response across full conversation context
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
                        # Save assistant response to state history
                        st.session_state.messages.append({"role": "assistant", "content": reply})
                    except Exception as e:
                        st.error(f"Error communicating with AI model: {e}")
