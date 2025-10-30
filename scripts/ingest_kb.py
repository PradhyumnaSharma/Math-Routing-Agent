
import os
import json
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "math_kb")
EMB_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

print("Using Qdrant at", QDRANT_URL)
client = QdrantClient(url=QDRANT_URL)
model = SentenceTransformer(EMB_MODEL)
DIM = model.get_sentence_embedding_dimension()

def is_valid_point_id(pid):
    """
    Qdrant accepts:
      - unsigned integer (we treat Python int >=0)
      - UUID string in canonical form (with hyphens)
    This helper checks those two cases.
    """
    # integer case
    if isinstance(pid, int):
        return pid >= 0
    # numeric string? allow it as integer
    if isinstance(pid, str) and pid.isdigit():
        try:
            return int(pid) >= 0
        except Exception:
            pass
    # UUID string case
    if isinstance(pid, str):
        try:
            _ = uuid.UUID(pid)
            return True
        except Exception:
            return False
    return False

def ensure_collection_exists(collection_name: str):
    # Check existing collections
    try:
        cols = client.get_collections().collections
        existing = [c.name for c in cols]
    except Exception:
        # older/newer qdrant-client variations: fallback to API call
        try:
            cols = client.get_collections()
            existing = [c.name for c in cols]
        except Exception:
            existing = []

    if collection_name in existing:
        print(f"Collection '{collection_name}' already exists.")
        return

    print(f"Creating collection '{collection_name}' with vector size {DIM} ...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE)
    )
    print("Collection created.")

def ingest():
    # ensure collection exists
    ensure_collection_exists(COLLECTION)

    with open("sample_kb.json", "r", encoding="utf-8") as f:
        docs = json.load(f)

    points = []
    for doc in docs:
        raw_id = doc.get("id", None)
        # compute vector from question + final_answer (same as before)
        text = doc.get("question", "") + "\n" + doc.get("final_answer", "")
        vec = model.encode(text).tolist()

        # determine point id acceptable to Qdrant
        if is_valid_point_id(raw_id):
            # keep numeric string as integer if needed
            if isinstance(raw_id, str) and raw_id.isdigit():
                pid = int(raw_id)
            else:
                pid = raw_id
        else:
            # generate uuid4 canonical string
            pid = str(uuid.uuid4())

        payload = {
            "original_id": raw_id,
            "question": doc.get("question"),
            "steps": doc.get("steps", []),
            "final_answer": doc.get("final_answer", "")
        }

        # Qdrant 'points' expect id, vector, payload
        points.append({"id": pid, "vector": vec, "payload": payload})

    # upsert points
    print(f"Upserting {len(points)} points into collection '{COLLECTION}' ...")
    client.upsert(collection_name=COLLECTION, points=points)
    print("Upsert completed.")

if __name__ == "__main__":
    ingest()
