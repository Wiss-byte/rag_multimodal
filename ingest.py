
"""
INGESTION DU TEXTE ET DES TABLEAUX PDF AVEC CLIP.

- Extraction PyMuPDF
- Chunking LlamaIndex
- Extraction des tableaux
- Embeddings CLIP 512D
- Normalisation L2
- Indexation PGVector
- Identifiant SHA-256
"""

import sys
import hashlib
import logging
from pathlib import Path

import fitz

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore

from clip_embeddings import (
    encode_texts,
    check_embedding,
    EMBEDDING_DIM,
)


# ============================================================
# CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "rag",
    "user": "rag",
    "password": "rag",
}

TEXT_TABLE = "text_chunks_clip"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# SHA-256
# ============================================================

def calculate_document_id(pdf_path: Path) -> str:
    """Calcule le SHA-256 du PDF."""

    sha256 = hashlib.sha256()

    with pdf_path.open("rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            sha256.update(block)

    return sha256.hexdigest()


# ============================================================
# PGVECTOR
# ============================================================

def create_vector_store(table_name: str):
    """Initialise un espace vectoriel PGVector."""

    return PGVectorStore.from_params(
        table_name=table_name,
        embed_dim=EMBEDDING_DIM,
        **DB_CONFIG,
    )


text_store = create_vector_store(TEXT_TABLE)


# ============================================================
# EXTRACTION PDF
# ============================================================

def extract_pdf_text(
    pdf_path: Path,
    document_id: str,
) -> list[Document]:
    """Extrait le texte de chaque page."""

    logger.info("Extraction du texte PDF...")

    documents = []

    with fitz.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if not text:
                continue

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "type": "text",
                        "page": page_number,
                        "source": pdf_path.name,
                        "document_id": document_id,
                    },
                )
            )

    logger.info("%s pages extraites.", len(documents))

    return documents


# ============================================================
# CHUNKING
# ============================================================

def split_documents(
    documents: list[Document],
) -> list[TextNode]:
    """Découpe les pages en fragments textuels."""

    if not documents:
        return []

    logger.info("Découpage en chunks...")

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    nodes = splitter.get_nodes_from_documents(
        documents,
        show_progress=True,
    )

    for index, node in enumerate(nodes):

        node.metadata["type"] = "text"

        document_id = node.metadata["document_id"]

        page = node.metadata["page"]

        # Identifiant déterministe
        raw_id = f"{document_id}:text:{page}:{index}"

        node.id_ = hashlib.sha256(
            raw_id.encode("utf-8")
        ).hexdigest()

    logger.info("%s chunks générés.", len(nodes))

    return nodes


# ============================================================
# EXTRACTION DES TABLEAUX
# ============================================================

def extract_tables(
    pdf_path: Path,
    document_id: str,
) -> list[TextNode]:
    """Extrait les tableaux et les convertit en Markdown."""

    logger.info("Extraction des tableaux...")

    table_nodes = []

    with fitz.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            try:
                tables = page.find_tables()

                for table_index, table in enumerate(
                    tables.tables,
                    start=1,
                ):

                    rows = table.extract()

                    if not rows:
                        continue

                    headers = [
                        str(cell or "").strip()
                        for cell in rows[0]
                    ]

                    if not headers:
                        continue

                    markdown = (
                        "| " + " | ".join(headers) + " |\n"
                    )

                    markdown += (
                        "| "
                        + " | ".join(["---"] * len(headers))
                        + " |\n"
                    )

                    for row in rows[1:]:

                        cells = [
                            str(cell or "").strip()
                            for cell in row
                        ]

                        cells = (
                            cells[:len(headers)]
                            + [""] * max(
                                0,
                                len(headers) - len(cells),
                            )
                        )

                        markdown += (
                            "| " + " | ".join(cells) + " |\n"
                        )

                    raw_id = (
                        f"{document_id}:table:"
                        f"{page_number}:{table_index}"
                    )

                    node = TextNode(
                        id_=hashlib.sha256(
                            raw_id.encode("utf-8")
                        ).hexdigest(),
                        text=markdown,
                        metadata={
                            "type": "table",
                            "page": page_number,
                            "source": pdf_path.name,
                            "document_id": document_id,
                            "table_index": table_index,
                        },
                    )

                    table_nodes.append(node)

            except Exception as error:
                logger.warning(
                    "Erreur tableau page %s : %s",
                    page_number,
                    error,
                )

    logger.info("%s tableaux extraits.", len(table_nodes))

    return table_nodes


# ============================================================
# EMBEDDINGS ET INDEXATION
# ============================================================

def index_nodes(nodes: list[TextNode]) -> None:
    """Calcule CLIP et stocke les nodes dans PGVector."""

    if not nodes:
        logger.warning("Aucun élément à indexer.")
        return

    logger.info(
        "Génération des embeddings CLIP : %s éléments...",
        len(nodes),
    )

    texts = [node.get_content() for node in nodes]

    embeddings = encode_texts(texts)

    for node, embedding in zip(nodes, embeddings):

        if not check_embedding(embedding):
            raise ValueError("Embedding CLIP invalide.")

        node.embedding = embedding

    # Réindexation idempotente des mêmes identifiants
    node_ids = [node.node_id for node in nodes]

    text_store.delete_nodes(node_ids=node_ids)

    # Les vecteurs sont déjà calculés.
    text_store.add(nodes)

    logger.info(
        "Indexation PGVector terminée : %s éléments.",
        len(nodes),
    )


def index_text_nodes(nodes: list[TextNode]) -> None:
    """Indexe les chunks textuels."""
    index_nodes(nodes)


def index_table_nodes(nodes: list[TextNode]) -> None:
    """Indexe les tableaux."""
    index_nodes(nodes)


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def ingest_pdf(pdf_file: str) -> dict:

    pdf_path = Path(pdf_file)

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF introuvable : {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Le fichier doit être un PDF.")

    logger.info("=" * 60)
    logger.info("INGESTION TEXTE ET TABLEAUX")
    logger.info("=" * 60)

    document_id = calculate_document_id(pdf_path)

    logger.info("SHA-256 : %s", document_id)

    documents = extract_pdf_text(
        pdf_path,
        document_id,
    )

    text_nodes = split_documents(documents)

    table_nodes = extract_tables(
        pdf_path,
        document_id,
    )

    index_text_nodes(text_nodes)

    index_table_nodes(table_nodes)

    summary = {
        "document": pdf_path.name,
        "document_id": document_id,
        "pages": len(documents),
        "chunks": len(text_nodes),
        "tables": len(table_nodes),
        "embedding_model": "CLIP ViT-B-32",
        "embedding_dimension": EMBEDDING_DIM,
        "vector_table": TEXT_TABLE,
    }

    logger.info("INGESTION TERMINEE")
    logger.info("%s", summary)

    return summary


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: uv run python ingest.py "
            "data/attention.pdf"
        )
        sys.exit(1)

    try:
        ingest_pdf(sys.argv[1])

    except Exception:
        logger.exception("Erreur pendant l'ingestion.")
        sys.exit(1)
