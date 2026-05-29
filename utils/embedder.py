from langchain_community.embeddings import HuggingFaceEmbeddings

# Model name used for generating embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
