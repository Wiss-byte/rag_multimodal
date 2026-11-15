
"""
INGESTION DES IMAGES - MULTIMODAL RAG

Fonctionnalites :
1. Extraction des images depuis un PDF.
2. Sauvegarde des images localement.
3. Generation des embeddings avec CLIP ViT-B-32.
4. Normalisation des embeddings.
5. Indexation dans PostgreSQL / PGVector.
6. Conservation des metadonnees.
7. Identification du document avec SHA-256.
"""

import sys
import hashlib
import logging

from pathlib import Path
from io import BytesIO

import pymupdf
from PIL import Image

from sentence_transformers import SentenceTransformer

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)

from llama_index.core.schema import TextNode

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

CLIP_MODEL_NAME = "clip-ViT-B-32"

CLIP_DIM = 512

IMAGE_DIR = Path("data/images")

IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
    Calcule l'identifiant unique SHA-256 du PDF.
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
# 4. CHARGEMENT DU MODELE CLIP
# ============================================================

logger.info("Chargement du modèle CLIP...")

clip_model = SentenceTransformer(
    CLIP_MODEL_NAME
)

logger.info("Modèle CLIP chargé.")


# ============================================================
# 5. INITIALISATION PGVECTOR
# ============================================================

image_store = PGVectorStore.from_params(
    table_name="image_clip",
    embed_dim=CLIP_DIM,
    **DB_CONFIG
)


# ============================================================
# 6. EXTRACTION ET SAUVEGARDE DES IMAGES
# ============================================================

def extract_images(pdf_path: Path, document_id: str):

    """
    Extrait les images intégrées au PDF
    et les sauvegarde dans data/images.
    """

    logger.info("Début de l'extraction des images...")

    pdf = pymupdf.open(pdf_path)

    image_nodes = []

    extracted_count = 0

    for page_number, page in enumerate(pdf, start=1):

        image_list = page.get_images(full=True)

        for image_index, image_info in enumerate(image_list):

            xref = image_info[0]

            try:

                # Extraction de l'image
                base_image = pdf.extract_image(xref)

                image_bytes = base_image["image"]

                image_ext = base_image["ext"]

                # Chargement avec Pillow
                image = Image.open(
                    BytesIO(image_bytes)
                ).convert("RGB")

                # Nom unique du fichier
                image_name = (
                    f"{document_id[:12]}_"
                    f"p{page_number}_"
                    f"img{image_index + 1}.png"
                )

                image_path = IMAGE_DIR / image_name

                # Sauvegarde au format PNG
                image.save(
                    image_path,
                    format="PNG"
                )

                # Génération embedding CLIP normalisé
                embedding = clip_model.encode(
                    image,
                    normalize_embeddings=True
                )

                # Création du noeud LlamaIndex
                node = TextNode(
                    text=(
                        f"Image extraite de la page "
                        f"{page_number} du document "
                        f"{pdf_path.name}"
                    ),

                    embedding=embedding.tolist(),

                    metadata={
                        "type": "image",
                        "page": page_number,
                        "source": pdf_path.name,
                        "image_path": image_path.as_posix(),
                        "document_id": document_id,
                        "image_index": image_index + 1,
                        "image_extension": image_ext,
                    }
                )

                image_nodes.append(node)

                extracted_count += 1

                logger.info(
                    "Image extraite : page %s, image %s",
                    page_number,
                    image_index + 1
                )

            except Exception as error:

                logger.warning(
                    "Erreur extraction image page %s : %s",
                    page_number,
                    error
                )

    pdf.close()

    logger.info(
        "%s images extraites.",
        extracted_count
    )

    return image_nodes


# ============================================================
# 7. INDEXATION DES EMBEDDINGS CLIP
# ============================================================

def index_images(image_nodes):

    """
    Indexe les embeddings CLIP pré-calculés
    dans PostgreSQL / PGVector.

    Aucun embedding OpenAI ou MiniLM n'est utilisé ici.
    """

    if not image_nodes:

        logger.warning(
            "Aucune image à indexer."
        )

        return

    logger.info(
        "Indexation des images dans PGVector..."
    )

    # Les embeddings CLIP sont déjà calculés.
    # On désactive le modèle d'embedding par défaut.
    storage_context = StorageContext.from_defaults(
        vector_store=image_store
    )

    VectorStoreIndex(
        image_nodes,
        storage_context=storage_context,
        embed_model="none",
        show_progress=True
    )

    logger.info(
        "Indexation des images terminée."
    )


# ============================================================
# 8. PIPELINE PRINCIPAL
# ============================================================

def ingest_images(pdf_file: str):

    pdf_path = Path(pdf_file)

    if not pdf_path.exists():

        raise FileNotFoundError(
            f"PDF introuvable : {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":

        raise ValueError(
            "Le fichier doit être un PDF."
        )

    logger.info("=" * 55)

    logger.info(
        "DEMARRAGE DE L'INGESTION DES IMAGES"
    )

    logger.info("=" * 55)

    # Etape 1 : calcul SHA-256
    document_id = calculate_document_id(
        pdf_path
    )

    logger.info(
        "Document ID : %s",
        document_id
    )

    # Etape 2 : extraction des images
    image_nodes = extract_images(
        pdf_path,
        document_id
    )

    # Etape 3 : indexation CLIP
    index_images(image_nodes)

    logger.info("=" * 55)

    logger.info(
        "INGESTION DES IMAGES TERMINEE"
    )

    logger.info(
        "Nombre d'images indexées : %s",
        len(image_nodes)
    )

    logger.info("=" * 55)


# ============================================================
# 9. EXECUTION
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "Usage : uv run python ingest_images.py "
            "data/attention.pdf"
        )

        sys.exit(1)

    ingest_images(
        sys.argv[1]
    )
