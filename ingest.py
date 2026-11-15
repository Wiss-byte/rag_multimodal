
"""
INGESTION PDF - MULTIMODAL RAG

Fonctionnalités :
1. Extraction du texte PDF.
2. Découpage en chunks.
3. Extraction des tableaux.
4. Embeddings textuels MiniLM.
5. Indexation dans PostgreSQL / PGVector.
6. Identification du document par SHA-256.
"""

import sys
import hashlib
import logging

from pathlib import Path

import fitz

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore


# ============================================================
# 1. CONFIGURATION
# ============================================================

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "rag",
    "user": "rag",
    "password": "rag",
}

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

EMBED_DIM = 384

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

IMAGE_DIR = Path("data/images")

IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# 3. IDENTIFICATION SHA-256
# ============================================================

def calculate_document_id(pdf_path: Path) -> str:
    """
    Calcule l'empreinte SHA-256 du document PDF.
    """

    sha256 = hashlib.sha256()

    with pdf_path.open("rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):

            sha256.update(block)

    return sha256.hexdigest()


# ============================================================
# 4. INITIALISATION DU MODELE D'EMBEDDING
# ============================================================

logger.info("Chargement du modèle MiniLM...")

embed_model = HuggingFaceEmbedding(
    model_name=EMBED_MODEL
)


# ============================================================
# 5. INITIALISATION DES BASES VECTORIELLES
# ============================================================

def create_vector_store(table_name: str):

    return PGVectorStore.from_params(
        table_name=table_name,
        embed_dim=EMBED_DIM,
        **DB_CONFIG
    )


text_store = create_vector_store("text_chunks")

caption_store = create_vector_store("image_captions")


# ============================================================
# 6. EXTRACTION DU TEXTE PDF
# ============================================================

def extract_pdf_text(pdf_path: Path, document_id: str):

    logger.info("Extraction du texte PDF...")

    pdf = fitz.open(pdf_path)

    documents = []

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
                }
            )
        )

    pdf.close()

    logger.info(
        "%s pages contenant du texte extraites.",
        len(documents)
    )

    return documents


# ============================================================
# 7. DECOUPAGE EN CHUNKS
# ============================================================

def split_documents(documents):

    logger.info("Découpage du texte en chunks...")

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    nodes = splitter.get_nodes_from_documents(
        documents,
        show_progress=True
    )

    for node in nodes:

        node.metadata["type"] = "text"

    logger.info("%s chunks générés.", len(nodes))

    return nodes


# ============================================================
# 8. EXTRACTION DES TABLEAUX
# ============================================================

def extract_tables(pdf_path: Path, document_id: str):

    logger.info("Extraction des tableaux...")

    pdf = fitz.open(pdf_path)

    table_nodes = []

    for page_number, page in enumerate(pdf, start=1):

        try:

            tables = page.find_tables()

            for table_index, table in enumerate(tables.tables):

                rows = table.extract()

                if not rows:
                    continue

                # Conversion du tableau en texte Markdown
                headers = rows[0]

                markdown = "| " + " | ".join(
                    str(cell or "") for cell in headers
                ) + " |\n"

                markdown += "| " + " | ".join(
                    ["---"] * len(headers)
                ) + " |\n"

                for row in rows[1:]:

                    markdown += "| " + " | ".join(
                        str(cell or "") for cell in row
                    ) + " |\n"

                table_nodes.append(
                    TextNode(
                        text=markdown,
                        metadata={
                            "type": "table",
                            "page": page_number,
                            "source": pdf_path.name,
                            "document_id": document_id,
                            "table_index": table_index,
                        }
                    )
                )

        except Exception as error:

            logger.warning(
                "Erreur page %s : %s",
                page_number,
                error
            )

    pdf.close()

    logger.info(
        "%s tableaux extraits.",
        len(table_nodes)
    )

    return table_nodes


# ============================================================
# 9. INDEXATION DES CHUNKS TEXTUELS
# ============================================================

def index_text_nodes(nodes):

    if not nodes:

        logger.warning("Aucun chunk textuel à indexer.")

        return

    logger.info("Indexation des chunks textuels...")

    storage_context = StorageContext.from_defaults(
        vector_store=text_store
    )

    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    logger.info("Indexation textuelle terminée.")


# ============================================================
# 10. INDEXATION DES TABLEAUX
# ============================================================

def index_table_nodes(nodes):

    if not nodes:

        logger.warning("Aucun tableau à indexer.")

        return

    logger.info("Indexation des tableaux...")

    storage_context = StorageContext.from_defaults(
        vector_store=text_store
    )

    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    logger.info("Indexation des tableaux terminée.")


# ============================================================
# 11. PIPELINE PRINCIPAL
# ============================================================

def ingest_pdf(pdf_path: str):

    path = Path(pdf_path)

    if not path.exists():

        raise FileNotFoundError(
            f"PDF introuvable : {path}"
        )

    if path.suffix.lower() != ".pdf":

        raise ValueError("Le fichier doit être un PDF.")

    logger.info("=" * 55)
    logger.info("DEMARRAGE DE L'INGESTION")
    logger.info("=" * 55)

    # Etape 1 : SHA-256
    document_id = calculate_document_id(path)

    logger.info("Document ID : %s", document_id)

    # Etape 2 : extraction texte
    documents = extract_pdf_text(
        path,
        document_id
    )

    # Etape 3 : découpage
    text_nodes = split_documents(documents)

    # Etape 4 : extraction tableaux
    table_nodes = extract_tables(
        path,
        document_id
    )

    # Etape 5 : indexation
    index_text_nodes(text_nodes)

    index_table_nodes(table_nodes)

    logger.info("=" * 55)
    logger.info("INGESTION TERMINEE")
    logger.info("Chunks : %s", len(text_nodes))
    logger.info("Tableaux : %s", len(table_nodes))
    logger.info("=" * 55)


# ============================================================
# 12. EXECUTION
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage : uv run python ingest.py chemin/vers/document.pdf"
        )

        sys.exit(1)

    ingest_pdf(sys.argv[1])
