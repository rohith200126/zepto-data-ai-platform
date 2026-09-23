from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / "chroma_db"
MODEL_DIR = BASE_DIR / "models" / "all-MiniLM-L6-v2"


# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

COLLECTION_NAME = "zepto_policies"


# ---------------------------------------------------------
# Global model
# ---------------------------------------------------------

_embedding_model = None


# ---------------------------------------------------------
# Load embedding model
# ---------------------------------------------------------

def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        print("Loading embedding model...")

        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Embedding model not found at:\n{MODEL_DIR}\n\n"
                "Please make sure the all-MiniLM-L6-v2 model "
                "exists inside support_assistant/models/"
            )

        _embedding_model = SentenceTransformer(
            str(MODEL_DIR),
            local_files_only=True
        )

        print("Embedding model loaded successfully.")

    return _embedding_model


# ---------------------------------------------------------
# Get Chroma collection
# ---------------------------------------------------------

def get_collection():
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
    except Exception:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    return collection


# ---------------------------------------------------------
# Build / update index
# ---------------------------------------------------------

def build_index():

    print("Loading embedding model...")

    model = get_embedding_model()

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # -----------------------------------------------------
    # Get existing collection or create it
    # -----------------------------------------------------

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )

        print("ChromaDB collection already exists.")

    except Exception:

        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        print("Created ChromaDB collection.")

    # -----------------------------------------------------
    # Read documents
    # -----------------------------------------------------

    documents = []
    ids = []

    for file_path in sorted(DOCS_DIR.glob("*.txt")):

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:
            continue

        documents.append(text)

        # Example:
        # doc_01.txt -> doc_01_chunk_01

        doc_name = file_path.stem

        ids.append(
            f"{doc_name}_chunk_01"
        )

    print(
        f"Creating embeddings for {len(documents)} documents..."
    )

    # -----------------------------------------------------
    # Create embeddings
    # -----------------------------------------------------

    embeddings = model.encode(
        documents,
        convert_to_numpy=True
    ).tolist()

    # -----------------------------------------------------
    # Avoid duplicate IDs
    # -----------------------------------------------------

    existing = collection.get()

    existing_ids = set(
        existing.get("ids", [])
    )

    new_documents = []
    new_embeddings = []
    new_ids = []

    for doc, embedding, doc_id in zip(
        documents,
        embeddings,
        ids
    ):

        if doc_id not in existing_ids:

            new_documents.append(doc)
            new_embeddings.append(embedding)
            new_ids.append(doc_id)

    # -----------------------------------------------------
    # Add documents
    # -----------------------------------------------------

    if new_documents:

        collection.add(
            documents=new_documents,
            embeddings=new_embeddings,
            ids=new_ids
        )

        print("Documents added to ChromaDB.")

    else:

        print("No new documents to add.")

    print(
        f"Indexed chunks: {collection.count()}"
    )

    print(
        f"ChromaDB location: {CHROMA_DIR}"
    )

    return collection


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    build_index()