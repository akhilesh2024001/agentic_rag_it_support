
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Header
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.rag.workflow import ask
from app.rag.vectorstore import add_documents
from app.services.ingestion import load_file, chunk_documents, SUPPORTED
from app.services.audit import write_audit


# Logging configuration
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")
settings = get_settings()


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


@router.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.app_name
    }


@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        logger.info("Received chat request: %s", payload.question)

        # Execute LangGraph workflow
        result = ask(payload.question)

        logger.info("LangGraph workflow completed successfully")

        # Save audit information
        write_audit(
            payload.question,
            result["source_used"],
            result.get("trace", [])
        )

        logger.info("Audit record saved successfully")

        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get(
                "current_query",
                payload.question
            )
        }

    except Exception as exc:
        # Print complete Python traceback in Render logs
        logger.exception("Chat endpoint failed")

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc


@router.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    x_admin_key: str = Header(default="")
):
    try:
        # Verify admin API key
        if x_admin_key != settings.admin_api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid admin key"
            )

        # Validate uploaded file
        suffix = Path(file.filename or "").suffix.lower()

        if suffix not in SUPPORTED:
            raise HTTPException(
                status_code=400,
                detail=f"Supported: {', '.join(sorted(SUPPORTED))}"
            )

        # Create upload directory
        upload_dir = Path(settings.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        filename = Path(file.filename).name
        dest = upload_dir / filename

        dest.write_bytes(await file.read())

        logger.info("File uploaded: %s", filename)

        # Load and split document
        docs = load_file(dest)
        chunks = chunk_documents(docs)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No text content found in the uploaded file"
            )

        # Add chunks to Pinecone
        ids = add_documents(chunks)

        logger.info(
            "Document indexed successfully: %s, chunks: %d",
            filename,
            len(chunks)
        )

        return {
            "message": "Document indexed",
            "file": dest.name,
            "chunks": len(chunks),
            "ids_created": len(ids)
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Document ingestion failed")

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc
