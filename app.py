import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
from utils.loader import extract_text, extract_text_from_url
from utils.retriever import build_vectorstore, retrieve_context

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("🤖 AI Chatbot with Document Reading")

# ── Sidebar ───────────────────────────────────────────
st.sidebar.header("🔑 Your API Key")
user_api_key = st.sidebar.text_input(
    "Enter your Groq API key",
    type="password",
    placeholder="gsk_..."
)

api_key = user_api_key or os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("Please enter your Groq API key in the sidebar.")
    st.stop()

client = Groq(api_key=api_key)

# ── File Upload ───────────────────────────────────────
st.sidebar.header("📄 Upload a Document")
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF, TXT, or Image",
    type=["pdf", "txt", "png", "jpg", "jpeg", "webp", "bmp"]
)

st.sidebar.header("🌐 Or Enter a URL")
url_input = st.sidebar.text_input("Paste a webpage URL", placeholder="https://...")
load_url_btn = st.sidebar.button("Load URL")

# ── Process sources ───────────────────────────────────
if uploaded_file:
    if "vectorstore" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
        with st.spinner("Reading and processing your document..."):
            text = extract_text(uploaded_file, client)
            if text.strip():
                st.session_state.vectorstore = build_vectorstore(text)
                st.session_state.last_file = uploaded_file.name
                st.sidebar.success("✅ Document ready!")
            else:
                st.sidebar.error("❌ Could not extract any content.")

elif load_url_btn and url_input:
    if st.session_state.get("last_file") != url_input:
        with st.spinner("Fetching and processing URL..."):
            text = extract_text_from_url(url_input)
            if text.strip() and not text.startswith("ERROR"):
                st.session_state.vectorstore = build_vectorstore(text)
                st.session_state.last_file = url_input
                st.sidebar.success("✅ URL content ready!")
            else:
                st.sidebar.error(f"❌ Could not extract content. {text}")

else:
    if "vectorstore" in st.session_state:
        del st.session_state.vectorstore

# ── Chat History ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat Input ────────────────────────────────────────
if prompt := st.chat_input("Ask me anything..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # If a document/image/URL is loaded, find relevant chunks
    if "vectorstore" in st.session_state:
        context = retrieve_context(st.session_state.vectorstore, prompt)
        system_prompt = f"""You are a helpful assistant.
Answer the user's question based on the document context below.
If the answer isn't in the document, say so honestly.

Document context:
{context}"""
    else:
        system_prompt = "You are a helpful assistant."

    # Build messages with system prompt
    messages_with_system = [
        {"role": "system", "content": system_prompt}
    ] + st.session_state.messages

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages_with_system,
        stream=True
    )

    with st.chat_message("assistant"):
        reply = ""
        placeholder = st.empty()
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            reply += content
            placeholder.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
