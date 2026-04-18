from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import streamlit as st

# Load model
llm = OllamaLLM(model="llama3")  # better than mistral

# ------------------------
# SESSION STATE INIT
# ------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_memory" not in st.session_state:
    st.session_state.user_memory = {}

# ------------------------
# FUNCTION
# ------------------------
def run_ai_chat(user_input):
    user_memory = st.session_state.user_memory
    chat_history = st.session_state.chat_history

    text = user_input.lower()

    # ------------------------
    # MEMORY EXTRACTION
    # ------------------------
    if "my name is" in text:
        name = user_input.split("is")[-1].strip()
        user_memory["name"] = name

    elif "save my name as" in text:
        name = user_input.split("as")[-1].strip()
        user_memory["name"] = name

    # ------------------------
    # SYSTEM PROMPT
    # ------------------------
    system_message = f"""
You are a smart, conversational AI assistant.

Known user info:
Name: {user_memory.get("name", "unknown")}

Rules:
- Be conversational
- Ask follow-up questions
- Use user's name if known
- Never say you don't remember the name if it's provided
"""

    # ------------------------
    # BUILD CHAT
    # ------------------------
    messages = [{"role": "system", "content": system_message}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_input})

    # ------------------------
    # MODEL CALL
    # ------------------------
    response = llm.invoke(messages)

    # ------------------------
    # SAVE HISTORY
    # ------------------------
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": response})

    return response

# ------------------------
# UI
# ------------------------
st.set_page_config(page_title="AI Chatbot", layout="centered")

st.title("🤖 AK - AI Chatbot")

# Display chat history
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    st.chat_message("user").write(user_input)

    response = run_ai_chat(user_input)

    st.chat_message("assistant").write(response)