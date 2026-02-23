import os 
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assets_dir = os.path.join(BASE_DIR, 'assets')
os.makedirs(assets_dir, exist_ok=True)

file_path = os.path.join(assets_dir, 'my_profile.txt')
CHROMA_DB_PATH = os.path.join(assets_dir, "chroma_db")
UPLOAD_DIR = os.path.join(assets_dir, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

EMBEDDINGS_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral:latest" #gemma3:4b

DEVICE = "cpu"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150