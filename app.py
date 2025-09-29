import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# -------------------------------
# 1. Load Model & Tokenizer
# -------------------------------
# Pick a local model (downloaded from Hugging Face)
# Example: "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
# Make sure you downloaded the model before running.
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"

st.set_page_config(page_title="ERIK - Local AI", page_icon="🤖", layout="wide")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto"
    )
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

generator = load_model()

# -------------------------------
# 2. System Prompt for ERIK
# -------------------------------
SYSTEM_PROMPT = """
You are ERIK (Exceptional Resources & Intelligence Kernel), 
an advanced AI assistant running locally. 
Your purpose is to provide clear, reliable, and deeply insightful answers.

Guidelines:
1. Be logical, concise, and human-like in your responses.
2. Break down complex topics into simple explanations.
3. Provide clean, optimized, and documented code for programming tasks.
4. Stay polite, professional, and unbiased.
5. If unsure, admit uncertainty rather than fabricating answers.
6. Always refer to yourself as ERIK.

Identity:
- Name: ERIK
- Meaning: Exceptional Resources & Intelligence Kernel
- Personality: Smart, supportive, reliable, futuristic but approachable.
- Goal: Empower the user with resources, intelligence, and knowledge.
"""

# -------------------------------
# 3. Streamlit UI
# -------------------------------
st.title("🤖 ERIK: Exceptional Resources & Intelligence Kernel")
st.write("Your personal AI assistant running locally without API.")

if "history" not in st.session_state:
    st.session_state["history"] = []

user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message
    st.session_state["history"].append({"role": "user", "content": user_input})

    # Generate AI response
    conversation = SYSTEM_PROMPT
    for msg in st.session_state["history"]:
        conversation += f"\n{msg['role'].capitalize()}: {msg['content']}"

    response = generator(conversation, max_new_tokens=400, temperature=0.7, do_sample=True)
    bot_reply = response[0]["generated_text"].split("User:")[-1].strip()

    st.session_state["history"].append({"role": "erik", "content": bot_reply})

# Display chat history
for msg in st.session_state["history"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])
