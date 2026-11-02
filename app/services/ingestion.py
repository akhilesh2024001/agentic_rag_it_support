
from pathlib import Path
from typing import Iterable

from docx import Document as DocxDocument
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# Supported document formats
SUPPORTED = {".pdf", ".txt", ".md", ".docx"}


def load_file(path: Path) -> list[Document]:
    """
    Load a file and convert its content into
    LangChain Document objects.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Document not found: {path}"
        )

    suffix = path.suffix.lower()

    # Store filename, not server filesystem path
    source = path.name

    # ---------------- PDF ----------------
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        documents = []

        for page_number, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source,
                            "page": page_number,
                        },
                    )
                )

        return documents

    # ---------------- TXT / MD ----------------
    if suffix in {".txt", ".md"}:
        text = path.read_text(
            encoding="utf-8-sig"
        )

        if not text.strip():
            return []

        return [
            Document(
                page_content=text,
                metadata={"source": source},
            )
        ]

    # ---------------- DOCX ----------------
    if suffix == ".docx":
        docx = DocxDocument(str(path))

        # Extract paragraphs
        paragraphs = [
            paragraph.text
            for paragraph in docx.paragraphs
            if paragraph.text.strip()
        ]

        # Extract tables
        for table in docx.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip()
                    for cell in row.cells
                )

                if row_text.strip():
                    paragraphs.append(row_text)

        text = "\n".join(paragraphs)

        if not text.strip():
            return []

        return [
            Document(
                page_content=text,
                metadata={"source": source},
            )
        ]

    raise ValueError(
        f"Unsupported file type: {suffix}"
    )


def chunk_documents(
    docs: Iterable[Document],
) -> list[Document]:
    """
    Split documents into smaller chunks
    for embedding and retrieval.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        add_start_index=True,
    )

    valid_docs = [
        document
        for document in docs
        if document.page_content.strip()
    ]

    if not valid_docs:
        return []

    chunks = splitter.split_documents(valid_docs)

    return chunks
