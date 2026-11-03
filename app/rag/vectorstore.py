
import logging
import time
from threading import Lock

from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings


# ==========================================
# 1. Configuration
# ==========================================

logger = logging.getLogger(__name__)

settings = get_settings()

EMBEDDING_DIMENSION = 384

# Shared objects (one instance per process)
_embeddings = None
_vectorstore = None
_index = None

# Thread-safe initialization locks
_embeddings_lock = Lock()
_index_lock = Lock()
_vectorstore_lock = Lock()


# ==========================================
# 2. Initialize FastEmbed
# ==========================================

def get_embeddings():
    global _embeddings

    if _embeddings is None:
        with _embeddings_lock:

            # Double-check after acquiring lock
            if _embeddings is None:

                logger.info(
                    "Initializing FastEmbed model: %s",
                    settings.embedding_model,
                )

                _embeddings = FastEmbedEmbeddings(
                    model_name=settings.embedding_model,
                    cache_dir=settings.embedding_cache_dir,
                    threads=1,
                    batch_size=16,
                )

                logger.info(
                    "FastEmbed model initialized successfully"
                )

    return _embeddings


# ==========================================
# 3. Connect to Pinecone
# ==========================================

def ensure_index():
    global _index

    if _index is None:
        with _index_lock:

            if _index is None:

                if not settings.pinecone_api_key:
                    raise RuntimeError(
                        "PINECONE_API_KEY is missing"
                    )

                logger.info(
                    "Connecting to Pinecone index: %s",
                    settings.pinecone_index_name,
                )

                pc = Pinecone(
                    api_key=settings.pinecone_api_key
                )

                # Get existing index names
                index_names = [
                    item["name"]
                    for item in pc.list_indexes()
                ]

                # Create index only if it does not exist
                if settings.pinecone_index_name not in index_names:

                    logger.info(
                        "Pinecone index does not exist. Creating it."
                    )

                    pc.create_index(
                        name=settings.pinecone_index_name,
                        dimension=EMBEDDING_DIMENSION,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud=settings.pinecone_cloud,
                            region=settings.pinecone_region,
                        ),
                    )

                    # Wait for index readiness
                    while True:

                        info = pc.describe_index(
                            settings.pinecone_index_name
                        )

                        if info.status["ready"]:
                            break

                        logger.info(
                            "Waiting for Pinecone index readiness..."
                        )

                        time.sleep(2)

                # Validate existing index dimension
                info = pc.describe_index(
                    settings.pinecone_index_name
                )

                current_dimension = getattr(
                    info,
                    "dimension",
                    None,
                )

                if current_dimension != EMBEDDING_DIMENSION:

                    raise RuntimeError(
                        "Pinecone dimension mismatch: "
                        f"Expected {EMBEDDING_DIMENSION}, "
                        f"found {current_dimension}. "
                        "Check your embedding model and index."
                    )

                # Connect to existing index
                _index = pc.Index(
                    settings.pinecone_index_name
                )

                logger.info(
                    "Connected to Pinecone successfully"
                )

    return _index


# ==========================================
# 4. Initialize Vector Store
# ==========================================

def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        with _vectorstore_lock:

            if _vectorstore is None:

                logger.info(
                    "Initializing Pinecone vector store"
                )

                _vectorstore = PineconeVectorStore(
                    index=ensure_index(),
                    embedding=get_embeddings(),
                    namespace=settings.pinecone_namespace,
                )

                logger.info(
                    "Vector store initialized successfully"
                )

    return _vectorstore


# ==========================================
# 5. Get Retriever
# ==========================================

def get_retriever():

    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={
            "k": settings.top_k
        }
    )


# ==========================================
# 6. Add Documents to Pinecone
# ==========================================

def add_documents(chunks):

    if not chunks:
        return []

    vectorstore = get_vectorstore()

    logger.info(
        "Adding %d document chunks to Pinecone",
        len(chunks),
    )

    ids = vectorstore.add_documents(chunks)

    logger.info(
        "Successfully indexed %d document chunks",
        len(ids),
    )

    return ids


# ==========================================
# 7. Check Pinecone Index Status
# ==========================================

def get_index_status():

    index = ensure_index()

    stats = index.describe_index_stats()

    return {
        "status": "ok",
        "index": settings.pinecone_index_name,
        "namespace": settings.pinecone_namespace,
        "dimension": stats.dimension,
        "total_vector_count": stats.total_vector_count,
    }
