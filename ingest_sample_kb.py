
"""
Ingest sample IT support knowledge-base documents
into the Pinecone vector database.

Run:
    python ingest_sample_kb.py
"""

from pathlib import Path

from app.core.config import get_settings
from app.services.ingestion import (
    SUPPORTED,
    load_file,
    chunk_documents,
)
from app.rag.vectorstore import add_documents


def main():

    # Step 1: Load application configuration
    settings = get_settings()

    # Step 2: Locate knowledge-base folder
    folder = Path(settings.sample_kb_dir)

    if not folder.is_dir():
        raise SystemExit(
            f"Knowledge-base folder not found: {folder}"
        )

    # Step 3: Find supported documents
    files = sorted(
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED
    )

    if not files:
        raise SystemExit(
            f"No supported documents found in {folder}"
        )

    print(f"Found {len(files)} knowledge-base files")

    # Step 4: Extract text from documents
    all_documents = []

    for path in files:
        print(f"Loading: {path.name}")

        documents = load_file(path)
        all_documents.extend(documents)

    if not all_documents:
        raise SystemExit(
            "Documents contain no extractable text."
        )

    # Step 5: Split into smaller chunks
    chunks = chunk_documents(all_documents)

    if not chunks:
        raise SystemExit(
            "No valid document chunks generated."
        )

    print(f"Generated {len(chunks)} chunks")

    # Step 6: Generate embeddings and upload to Pinecone
    print("Uploading embeddings to Pinecone...")

    ids = add_documents(chunks)

    # Step 7: Report successful ingestion
    print("\nKnowledge-base ingestion completed!")
    print(f"Files processed: {len(files)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Vectors uploaded: {len(ids)}")
    print(f"Pinecone index: {settings.pinecone_index_name}")
    print(f"Namespace: {settings.pinecone_namespace}")


if __name__ == "__main__":
    main()
