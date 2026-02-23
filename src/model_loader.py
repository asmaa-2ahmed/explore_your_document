from langchain_ollama import OllamaLLM
from .config import OLLAMA_MODEL

_llm_instance = None

def load_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = OllamaLLM(model=OLLAMA_MODEL, temperature=0.7)
    return _llm_instance