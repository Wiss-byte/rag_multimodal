
"""
INGESTION DES IMAGES PDF AVEC CLIP.

- Extraction PyMuPDF
- Sauvegarde PNG
- Embeddings CLIP 512D
- Normalisation L2
- Indexation PGVector
- Métadonnées SHA-256
"""

import sys
import hashlib
import logging

from pathlib import Path
from io import BytesIO

import fitz
from PIL import Image

from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore

from clip_embeddings import (
    encode_image_object,
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

IMAGE_TABLE = "image_clip"

IMAGE_DIR = Path("data/images")
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


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

image_store = PGVectorStore.from_params(
    table_name=IMAGE_TABLE,
    embed_dim=EMBEDDING_DIM,
    **DB_CONFIG,
)


# ============================================================
# EXTRACTION DES IMAGES
# ============================================================

def extract_images(
    pdf_path: Path,
    document_id: str,
) -> list[TextNode]:

    logger.info("Extraction des images PDF...")

    image_nodes = []

    with fitz.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf, start=1):

            image_list = page.get_images(full=True)

            for image_index, image_info in enumerate(
                image_list,
                start=1,
            ):

                xref = image_info[0]

                try:

                    base_image = pdf.extract_image(xref)

                    image_bytes = base_image["image"]

                    image_ext = base_image["ext"]

                    image = Image.open(
                        BytesIO(image_bytes)
                    ).convert("RGB")

                    image_name = (
                        f"{document_id[:12]}_"
                        f"p{page_number}_"
                        f"img{image_index}.png"
                    )

                    image_path = IMAGE_DIR / image_name

                    image.save(image_path, format="PNG")

                    # Embedding CLIP
                    embedding = encode_image_object(image)

                    if not check_embedding(embedding):
                        raise ValueError(
                            "Embedding image invalide."
                        )

                    raw_id = (
                        f"{document_id}:image:"
                        f"{page_number}:{image_index}"
                    )

                    node = TextNode(
                        id_=hashlib.sha256(
                            raw_id.encode("utf-8")
                        ).hexdigest(),
                        text=(
                            f"Image du document {pdf_path.name}, "
                            f"page {page_number}"
                        ),
                        embedding=embedding,
                        metadata={
                            "type": "image",
                            "page": page_number,
                            "source": pdf_path.name,
                            "document_id": document_id,
                            "image_path": image_path.as_posix(),
                            "image_index": image_index,
                            "image_extension": image_ext,
                        },
                    )

                    image_nodes.append(node)

                    logger.info(
                        "Image extraite : page %s, image %s",
                        page_number,
                        image_index,
                    )

                except Exception as error:

                    logger.warning(
                        "Erreur image page %s : %s",
                        page_number,
                        error,
                    )

    logger.info("%s images extraites.", len(image_nodes))

    return image_nodes


# ============================================================
# INDEXATION
# ============================================================

def index_images(image_nodes: list[TextNode]) -> None:

    if not image_nodes:
        logger.warning("Aucune image à indexer.")
        return

    logger.info("Indexation des images CLIP...")

    # Les embeddings ont déjà été calculés.
    # On évite tout recalcul avec un autre modèle.

    node_ids = [node.node_id for node in image_nodes]

    image_store.delete_nodes(node_ids=node_ids)

    image_store.add(image_nodes)

    logger.info(
        "%s images indexées dans PGVector.",
        len(image_nodes),
    )


# ============================================================
# PIPELINE PRINCIPAL
# ============================================================

def ingest_images(pdf_file: str) -> dict:

    pdf_path = Path(pdf_file)

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF introuvable : {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Le fichier doit être un PDF.")

    document_id = calculate_document_id(pdf_path)

    logger.info("Document SHA-256 : %s", document_id)

    image_nodes = extract_images(
        pdf_path,
        document_id,
    )

    index_images(image_nodes)

    return {
        "document": pdf_path.name,
        "document_id": document_id,
        "images": len(image_nodes),
        "embedding_model": "CLIP ViT-B-32",
        "embedding_dimension": EMBEDDING_DIM,
        "vector_table": IMAGE_TABLE,
    }


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Usage: uv run python ingest_images.py "
            "data/attention.pdf"
        )
        sys.exit(1)

    try:
        result = ingest_images(sys.argv[1])
        print(result)

    except Exception:
        logger.exception("Erreur ingestion images.")
        sys.exit(1)
