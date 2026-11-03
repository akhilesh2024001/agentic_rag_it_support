
import time

from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings

settings = get_settings()

_embeddings = None
_vectorstore = None
_index = None

EMBEDDING_DIMENSION = 384


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(
            model_name=settings.embedding_model,
            cache_dir=settings.embedding_cache_dir,
            threads=1,
            batch_size=16,
        )

    return _embeddings


def ensure_index():
    global _index

    if _index is not None:
        return _index

    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    pc = Pinecone(api_key=settings.pinecone_api_key)

    index_names = [
        item["name"] for item in pc.list_indexes()
    ]

    if settings.pinecone_index_name in index_names:
        info = pc.describe_index(settings.pinecone_index_name)
        current_dimension = getattr(info, "dimension", None)

        if current_dimension != EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Pinecone dimension mismatch: "
                f"{current_dimension} != {EMBEDDING_DIMENSION}"
            )

    else:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

        while not pc.describe_index(
            settings.pinecone_index_name
        ).status["ready"]:
            time.sleep(1)

    _index = pc.Index(settings.pinecone_index_name)

    return _index


def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        _vectorstore = PineconeVectorStore(
            index=ensure_index(),
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


def get_retriever():
    return get_vectorstore().as_retriever(
        search_kwargs={"k": settings.top_k}
    )


def add_documents(chunks):
    return get_vectorstore().add_documents(chunks)
