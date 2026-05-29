from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.embedder import get_embeddings

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def split_text(text: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_text(text)


def build_vectorstore(text: str):
    chunks = split_text(text)
    embeddings = get_embeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore


def retrieve_context(vectorstore, query: str, k: int = 3) -> str:
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])
