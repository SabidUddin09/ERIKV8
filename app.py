import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# -------------------------------
# SETTINGS
# -------------------------------
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"  # Change if you want another model
SYSTEM_PROMPT = """
You are ERIK (Exceptional Resources & Intelligence Kernel), 
an advanced AI assistant. 
You are smart, supportive, reliable, futuristic but approachable. 
You always give clear, logical, and human-like answers. 
You can act as a teacher, coder, researcher, or advisor as per user needs. 
If unsure, admit uncertainty instead of making things up.
"""

st.set_page_config(page_title="ERIK - Local Chatbot", page_icon="🤖")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",    # uses GPU if available
        torch_dtype="auto"
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

generator = load_model()

# -------------------------------
# CHAT INTERFACE
# -------------------------------
st.title("🤖 ERIK: Exceptional Resources & Intelligence Kernel")
st.write("Your personal AI assistant running locally without API.")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state["history"] = [{"role": "system", "content": SYSTEM_PROMPT}]

# User input box
user_input = st.chat_input("Type your message here...")

if user_input:
    # Save user message
    st.session_state["history"].append({"role": "user", "content": user_input})

    # Build conversation string
    conversation = ""
    for msg in st.session_state["history"]:
        if msg["role"] != "system":
            conversation += f"\n{msg['role'].capitalize()}: {msg['content']}"

    # Generate response
    response = generator(
        SYSTEM_PROMPT + conversation,
        max_new_tokens=400,
        temperature=0.7,
        do_sample=True,
    )
    bot_reply = response[0]["generated_text"].split("User:")[-1].strip()

    # Save ERIK's reply
    st.session_state["history"].append({"role": "erik", "content": bot_reply})

# Display chat
for msg in st.session_state["history"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    elif msg["role"] == "erik":
        st.chat_message("assistant").write(msg["content"])
