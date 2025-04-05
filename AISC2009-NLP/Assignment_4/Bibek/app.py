import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load Groq API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Groq client
client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")

# Prompt Strategies
def few_shot_prompt(user_input):
    return f"""You are a helpful assistant. Respond appropriately.

User: How can I reset my password?
Assistant: You can reset your password by clicking on 'Forgot Password' on the login screen.

User: I’m unable to log into my account after resetting the password.
Assistant: Please ensure you're using the most recent password. If the issue persists, contact our support team.

User: {user_input}
Assistant:"""

def cot_prompt(user_input):
    return f"You are a thoughtful assistant. Think step by step to help users.\nUser: {user_input}\nAssistant: Let's think step by step."

# Groq Response Generator
def generate_response(prompt, model="llama3-8b-8192"):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
        temperature=0.7,
        top_p=1,
    )
    return response.choices[0].message.content.strip()

# Streamlit Layout
st.set_page_config(page_title="Prompt Engineering Chatbot", layout="centered")
st.title("💬 Prompt Engineering Chatbot")

# Session state to hold history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User Input
st.subheader("🧠 Ask a Question")
user_input = st.text_input("Your query:")
mode = st.selectbox("Choose Prompt Strategy", ["Few-Shot", "Chain-of-Thought (CoT)"])

if st.button("Send") and user_input:
    if mode == "Few-Shot":
        prompt = few_shot_prompt(user_input)
    else:
        prompt = cot_prompt(user_input)

    with st.spinner("Thinking..."):
        response = generate_response(prompt)

    # Store in session
    st.session_state.chat_history.append((user_input, response))

# Display history
st.subheader("🗂️ Chat History")
for idx, (q, a) in enumerate(reversed(st.session_state.chat_history[-5:])):  # last 5 messages
    st.markdown(f"**User:** {q}")
    st.markdown(f"**Assistant:** {a}")
    st.markdown("---")
