
"""Pinecone vector database and FastEmbed integration."""

import hashlib
import logging
import math
import time

from langchain_core.embeddings import Embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings


logger = logging.getLogger(__name__)
settings = get_settings()

_embeddings = None
_vectorstore = None


# MiniLM generates 384-dimensional vectors
def get_embedding_dimension(model_name: str | None = None) -> int:
    model = (
        model_name or settings.embedding_model
    ).lower().strip()

    supported = (
        "all-minilm-l6-v2",
        "sentence-transformers/all-minilm-l6-v2",
    )

    if model not in supported:
        raise ValueError(
            f"Unsupported embedding model: {model}. "
            "Only MiniLM-L6-v2 is supported."
        )

    return 384


# Embedding implementation using FastEmbed (ONNX)
class MiniLMEmbeddings(Embeddings):

    def __init__(self):
        from fastembed import TextEmbedding

        self.model = TextEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=settings.embedding_cache_dir,
        )

    @staticmethod
    def _normalize(vector):
        values = [float(x) for x in vector]

        magnitude = math.sqrt(
            sum(x * x for x in values)
        ) or 1.0

        return [x / magnitude for x in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors = self.model.embed(
            texts,
            batch_size=16,
        )

        return [self._normalize(v) for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


# Initialize embeddings only when needed
def get_embeddings():
    global _embeddings

    if _embeddings is None:
        get_embedding_dimension()
        _embeddings = MiniLMEmbeddings()

    return _embeddings


# Initialize Pinecone client
def _pinecone_client():
    if not settings.pinecone_api_key:
        raise RuntimeError(
            "PINECONE_API_KEY is missing. "
            "Configure it in your environment."
        )

    return Pinecone(api_key=settings.pinecone_api_key)


# Create or validate Pinecone index
def ensure_index():
    pc = _pinecone_client()

    indexes = pc.list_indexes()

    if hasattr(indexes, "names"):
        names = set(indexes.names())
    else:
        names = {i["name"] for i in indexes}

    index_name = settings.pinecone_index_name

    if index_name not in names:
        logger.info(
            "Creating Pinecone index: %s",
            index_name,
        )

        pc.create_index(
            name=index_name,
            dimension=get_embedding_dimension(),
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

        # Wait for new index to become ready
        for _ in range(60):
            info = pc.describe_index(index_name)

            if info.status["ready"]:
                break

            time.sleep(2)
        else:
            raise RuntimeError(
                "Pinecone index did not become ready."
            )

    # Validate existing index dimensions
    info = pc.describe_index(index_name)

    dimension = getattr(info, "dimension", None)

    if dimension is None and isinstance(info, dict):
        dimension = info.get("dimension")

    if int(dimension or 0) != get_embedding_dimension():
        raise RuntimeError(
            f"Pinecone index dimension is {dimension}. "
            "MiniLM requires dimension 384. "
            "Use a matching Pinecone index."
        )

    return pc.Index(index_name)


# Check index status without modifying it
def get_index_status():
    pc = _pinecone_client()

    indexes = pc.list_indexes()

    if hasattr(indexes, "names"):
        names = indexes.names()
    else:
        names = [i["name"] for i in indexes]

    index_name = settings.pinecone_index_name

    if index_name not in names:
        return {
            "status": "missing_index",
            "vectors_in_namespace": 0,
        }

    index = pc.Index(index_name)
    stats = index.describe_index_stats()

    namespaces = getattr(stats, "namespaces", {}) or {}

    namespace = namespaces.get(
        settings.pinecone_namespace
    )

    count = getattr(namespace, "vector_count", None)

    if count is None and isinstance(namespace, dict):
        count = namespace.get("vector_count", 0)

    dimension = getattr(stats, "dimension", None)

    if dimension != get_embedding_dimension():
        status = "dimension_mismatch"
    elif not count:
        status = "empty"
    else:
        status = "ok"

    return {
        "status": status,
        "vectors_in_namespace": int(count or 0),
        "dimension": dimension,
    }


# Connect LangChain to Pinecone
def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        _vectorstore = PineconeVectorStore(
            index=ensure_index(),
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


# Retrieve relevant documents
def get_retriever():
    return get_vectorstore().as_retriever(
        search_kwargs={
            "k": settings.top_k
        }
    )


# Store document chunks in Pinecone
def add_documents(chunks):
    if not chunks:
        return []

    # Stable IDs prevent identical chunks from being duplicated
    ids = []

    for doc in chunks:
        content = (
            doc.metadata.get("source", "")
            + "|"
            + str(doc.metadata.get("page", ""))
            + "|"
            + str(doc.metadata.get("start_index", 0))
            + "|"
            + doc.page_content
        )

        vector_id = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

        ids.append(vector_id)

    return get_vectorstore().add_documents(
        chunks,
        ids=ids,
    )
