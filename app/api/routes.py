
import hmac
import logging
from pathlib import Path

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
    Header,
)

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.rag.workflow import ask

from app.rag.vectorstore import (
    add_documents,
    get_index_status,
    get_embeddings,
)

from app.services.ingestion import (
    load_file,
    chunk_documents,
    SUPPORTED,
)

from app.services.audit import write_audit


# ==========================================
# 1. Application Configuration
# ==========================================

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

settings = get_settings()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


# ==========================================
# 2. Chat Request Schema
# ==========================================

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=3000,
    )


# ==========================================
# 3. Admin Authentication
# ==========================================

def require_admin(key: str):

    # Admin key must be configured securely
    if (
        not settings.admin_api_key
        or settings.admin_api_key == "change-me"
    ):
        raise HTTPException(
            status_code=503,
            detail="ADMIN_API_KEY must be configured securely",
        )

    # Secure key comparison
    if not hmac.compare_digest(
        key,
        settings.admin_api_key,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin key",
        )


# ==========================================
# 4. Health Check API
# ==========================================

@router.get("/health")
def health():

    # This confirms FastAPI is running.
    # It does not verify external API connectivity.

    return {
        "status": "alive",
        "service": settings.app_name,
    }


# ==========================================
# 5. Pipeline Diagnostics
# ==========================================

@router.get("/diagnostics")
def diagnostics(
    x_admin_key: str = Header(default=""),
    deep: bool = False,
):

    # Only an authorized admin can inspect deployment
    require_admin(x_admin_key)

    kb_folder = Path(settings.sample_kb_dir)

    result = {
        "configured": {
            "groq": bool(settings.groq_api_key),
            "pinecone": bool(settings.pinecone_api_key),
            "tavily_optional": bool(settings.tavily_api_key),
            "web_fallback_enabled": settings.web_fallback_enabled,
        },
        "index": settings.pinecone_index_name,
        "namespace": settings.pinecone_namespace,
        "sample_documents_on_disk": (
            len([
                p for p in kb_folder.iterdir()
                if p.is_file()
                and p.suffix.lower() in SUPPORTED
            ])
            if kb_folder.is_dir()
            else 0
        ),
        "pinecone": {
            "status": "not_checked"
        },
    }

    # Check Pinecone connectivity and vectors
    if settings.pinecone_api_key:

        try:
            result["pinecone"] = get_index_status()

        except Exception as exc:

            logger.exception(
                "Pinecone diagnostic failed"
            )

            result["pinecone"] = {
                "status": "error",
                "error_type": type(exc).__name__,
            }

    else:
        result["pinecone"] = {
            "status": "missing_api_key"
        }

    # Optional deep embedding verification
    if deep:

        try:
            embedding = get_embeddings()

            vector = embedding.embed_query(
                "IT support deployment diagnostic"
            )

            result["embeddings"] = {
                "status": "ok",
                "dimensions": len(vector),
            }

        except Exception as exc:

            logger.exception(
                "Embedding diagnostic failed"
            )

            result["embeddings"] = {
                "status": "error",
                "error_type": type(exc).__name__,
            }

    # Determine basic KB readiness
    embeddings_ok = result.get(
        "embeddings",
        {"status": "ok"},
    )["status"] == "ok"

    result["ready_for_kb_questions"] = (
        bool(settings.groq_api_key)
        and result["pinecone"].get("status") == "ok"
        and embeddings_ok
    )

    return result


# ==========================================
# 6. Main Agentic RAG Chat API
# ==========================================

@router.post("/chat")
def chat(payload: ChatRequest):

    try:

        # Execute the complete LangGraph pipeline
        result = ask(payload.question)

    except Exception as exc:

        logger.exception(
            "Agentic RAG pipeline execution failed"
        )

        # Return a useful response for missing keys
        if isinstance(exc, RuntimeError):

            message = str(exc)

            if message.startswith("GROQ_API_KEY is missing"):
                raise HTTPException(
                    status_code=503,
                    detail="GROQ_API_KEY is missing",
                ) from exc

            if message.startswith("PINECONE_API_KEY is missing"):
                raise HTTPException(
                    status_code=503,
                    detail="PINECONE_API_KEY is missing",
                ) from exc

        # Do not expose internal errors or credentials
        raise HTTPException(
            status_code=502,
            detail=(
                "Agent pipeline failed. "
                "Check deployment logs and "
                "the /api/diagnostics endpoint."
            ),
        ) from exc

    # Audit failure must not interrupt user response
    try:

        write_audit(
            payload.question,
            result["source_used"],
            result.get("trace", []),
        )

    except Exception:

        logger.exception(
            "Audit logging failed"
        )

    # Response format expected by frontend app.js
    return {
        "answer": result["answer"],
        "source_used": result["source_used"],
        "trace": result.get("trace", []),
        "citations": result.get("citations", []),
        "rewritten_query": result.get(
            "current_query",
            payload.question,
        ),
    }


# ==========================================
# 7. Document Upload and Ingestion API
# ==========================================

@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_admin_key: str = Header(default=""),
):

    # Verify administrator permissions
    require_admin(x_admin_key)

    filename = Path(
        file.filename or ""
    ).name

    # Validate supported file extensions
    if (
        not filename
        or filename.startswith(".")
        or Path(filename).suffix.lower() not in SUPPORTED
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Supported file types: "
                + ", ".join(sorted(SUPPORTED))
            ),
        )

    # Store uploads outside publicly served static folder
    upload_dir = Path(settings.upload_dir)

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = upload_dir / filename

    try:

        # Save uploaded document in limited-size chunks
        size = 0

        with destination.open("wb") as output:

            while True:

                part = await file.read(
                    1024 * 1024
                )

                if not part:
                    break

                size += len(part)

                if size > MAX_UPLOAD_BYTES:

                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Maximum document size "
                            "is 10 MB"
                        ),
                    )

                output.write(part)

        # Extract document text
        documents = load_file(destination)

        # Split document into chunks
        chunks = chunk_documents(documents)

        if not chunks:

            raise HTTPException(
                status_code=422,
                detail=(
                    "No extractable text found "
                    "in the uploaded document"
                ),
            )

        # Generate embeddings and store in Pinecone
        ids = add_documents(chunks)

    except HTTPException:

        destination.unlink(
            missing_ok=True
        )

        raise

    except Exception as exc:

        logger.exception(
            "Document ingestion failed"
        )

        destination.unlink(
            missing_ok=True
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Document ingestion failed. "
                "Check server logs. "
                f"Error type: {type(exc).__name__}"
            ),
        ) from exc

    finally:

        await file.close()

    return {
        "message": "Document indexed successfully",
        "file": filename,
        "chunks": len(chunks),
        "ids_created": len(ids),
    }
