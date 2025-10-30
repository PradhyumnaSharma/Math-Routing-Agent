from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
load_dotenv()

EMB_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMB_MODEL)
    return _model

def embed_text(text):
    model = get_model()
    vec = model.encode(text)
    return vec.tolist()
