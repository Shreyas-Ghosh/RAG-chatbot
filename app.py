import streamlit as st
from groq import Groq
import os
import base64
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pypdf import PdfReader

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

# ── Helper: Encode image to base64 ───────────────────
def encode_image(file) -> str:
    file.seek(0)
    return base64.b64encode(file.read()).decode("utf-8")

# ── Extract text from uploaded file ──────────────────
def extract_text(file):
    if file.name.endswith(".pdf"):
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text

    elif file.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        # Send image to Groq vision model for understanding
        image_data = encode_image(file)
        extension = file.name.rsplit(".", 1)[-1].lower()
        media_type = "jpeg" if extension in ("jpg", "jpeg") else extension

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{media_type};base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": """Describe this image in as much detail as possible.
Include: main subjects, colors, objects, setting, mood, any visible text,
and anything else notable. Be thorough — this description will be used
to answer questions about the image."""
                        }
                    ]
                }
            ],
            max_tokens=1024
        )
        return response.choices[0].message.content

    else:  # .txt
        return file.read().decode("utf-8")

# ── Extract text from a URL ───────────────────────────
def extract_text_from_url(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove noise: scripts, styles, nav, footers
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        # Clean up excessive blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to fetch URL: {e}")
        return ""

# ── Build vector store from text ─────────────────────
def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# ── Process sources ───────────────────────────────────
if uploaded_file:
    if "vectorstore" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
        with st.spinner("Reading and processing your document..."):
            text = extract_text(uploaded_file)
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
            if text.strip():
                st.session_state.vectorstore = build_vectorstore(text)
                st.session_state.last_file = url_input
                st.sidebar.success("✅ URL content ready!")
            else:
                st.sidebar.error("❌ No content could be extracted from that URL.")

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
        docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
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