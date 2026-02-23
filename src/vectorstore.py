import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from .config import CHROMA_DB_PATH, EMBEDDINGS_MODEL_ID

_embeddings = None
_vectorstore = None

def load_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_MODEL_ID
        )
    return _embeddings


def load_vectorstore(chunks=None):
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    embeddings = load_embeddings()

    if os.path.exists(CHROMA_DB_PATH):
        print("📂 Loading existing DB...")
        _vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embeddings
        )
    else:
        if chunks is None:
            raise ValueError("Chunks required to create DB.")

        print("🆕 Creating DB...")
        _vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        _vectorstore.persist()

    return _vectorstore