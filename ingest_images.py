"""
INGESTION DES IMAGES PDF AVEC CLIP.

- Extraction PyMuPDF
- Sauvegarde PNG
- Légende « Figure N: ... » rattachée à chaque image
- Description visuelle optionnelle par un VLM (Ollama)
- Embeddings CLIP 512D (image)
- Normalisation L2
- Indexation PGVector
- Identifiant SHA-256 + réindexation propre par document
"""

import re
import sys
import hashlib
import logging

from pathlib import Path
from io import BytesIO

import fitz
from PIL import Image

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
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

# Clé de métadonnée du document (ne pas utiliser "document_id" :
# LlamaIndex l'écrase lors de l'enregistrement dans PGVector)
DOC_KEY = "doc_sha256"

# Légendes de figures : « Figure 1: ... »
FIGURE_CAPTION_PATTERN = re.compile(r"^Figure\s+\d+\s*:")

# Images trop petites ignorées (icônes, logos)
MIN_IMAGE_SIDE = 32

POSITION_FR = {
    "left": "gauche",
    "right": "droite",
}

# Description visuelle par un VLM local (facultatif)
USE_VLM_DESCRIPTION = False
VLM_MODEL = "moondream"


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

image_store = PGVectorStore.from_params(
    table_name=IMAGE_TABLE,
    embed_dim=EMBEDDING_DIM,
    **DB_CONFIG,
)


def delete_document(document_id: str) -> None:
    """Supprime toutes les images d'un document avant réindexation."""

    image_store.delete_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key=DOC_KEY,
                    value=document_id,
                )
            ]
        )
    )

    logger.info(
        "Anciennes images du document supprimées : %s",
        document_id[:12],
    )


# ============================================================
# LEGENDES ET DESCRIPTIONS
# ============================================================

def find_figure_caption(page, image_rect) -> str:
    """Retrouve la légende « Figure N: ... » placée sous l'image."""

    caption_block = None

    for block in page.get_text("blocks"):

        y0 = block[1]
        text = block[4].strip()

        if not FIGURE_CAPTION_PATTERN.match(text):
            continue

        # La légende doit se trouver sous l'image
        if y0 < image_rect.y1 - 5:
            continue

        # On garde la légende la plus proche de l'image
        if caption_block is None or y0 < caption_block[1]:
            caption_block = block

    if caption_block is None:
        return ""

    return " ".join(caption_block[4].split())


def extract_sub_caption(caption: str, position: str) -> str:
    """
    Extrait la partie de légende propre à l'image :
    « (left) ... » ou « (right) ... ».
    """

    if not position:
        return ""

    match = re.search(
        rf"\({position}\)\s*(.+?)(?=\((?:left|right)\)|$)",
        caption,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return match.group(1).strip() if match else ""


def describe_image(image_path: Path) -> str:
    """Description courte de l'image par un VLM local (facultatif)."""

    if not USE_VLM_DESCRIPTION:
        return ""

    try:
        import ollama

        response = ollama.chat(
            model=VLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Describe this figure from a scientific paper "
                        "in 2 or 3 factual sentences."
                    ),
                    "images": [str(image_path)],
                }
            ],
        )

        return response.message.content.strip()

    except Exception as error:

        logger.warning(
            "Description VLM impossible pour %s : %s",
            image_path.name,
            error,
        )

        return ""


def build_image_text(
    page_number: int,
    caption: str,
    position: str,
    sub_caption: str,
    description: str,
) -> str:
    """Texte associé à l'image, lu par le LLM via l'outil de recherche."""

    lines = [
        caption or f"Figure sans légende (page {page_number})"
    ]

    if sub_caption:
        lines.append(
            f"Partie {POSITION_FR[position]} de la figure : {sub_caption}"
        )

    if description:
        lines.append(f"Description visuelle : {description}")

    return "\n".join(lines)


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

            page_center = page.rect.width / 2

            # 1er passage : position et légende de chaque image
            candidates = []

            for image_index, image_info in enumerate(
                page.get_images(full=True),
                start=1,
            ):

                xref = image_info[0]

                rects = page.get_image_rects(xref)

                rect = rects[0] if rects else None

                caption = (
                    find_figure_caption(page, rect)
                    if rect is not None
                    else ""
                )

                candidates.append({
                    "index": image_index,
                    "xref": xref,
                    "rect": rect,
                    "caption": caption,
                })

            # 2e passage : extraction, embedding, création des nodes
            for candidate in candidates:

                image_index = candidate["index"]
                rect = candidate["rect"]
                caption = candidate["caption"]

                try:

                    # Plusieurs images sous la même légende -> gauche / droite
                    shared = [
                        other for other in candidates
                        if caption and other["caption"] == caption
                    ]

                    position = ""

                    if len(shared) > 1 and rect is not None:
                        x_center = (rect.x0 + rect.x1) / 2
                        position = (
                            "left" if x_center < page_center else "right"
                        )

                    sub_caption = extract_sub_caption(caption, position)

                    base_image = pdf.extract_image(candidate["xref"])

                    image = Image.open(
                        BytesIO(base_image["image"])
                    ).convert("RGB")

                    if min(image.size) < MIN_IMAGE_SIDE:
                        logger.info(
                            "Image ignorée (trop petite) : page %s",
                            page_number,
                        )
                        continue

                    image_name = (
                        f"{document_id[:12]}_"
                        f"p{page_number}_"
                        f"img{image_index}.png"
                    )

                    image_path = IMAGE_DIR / image_name

                    image.save(image_path, format="PNG")

                    # Embedding CLIP de l'image
                    embedding = encode_image_object(image)

                    if not check_embedding(embedding):
                        raise ValueError(
                            "Embedding image invalide."
                        )

                    description = describe_image(image_path)

                    raw_id = (
                        f"{document_id}:image:"
                        f"{page_number}:{image_index}"
                    )

                    node = TextNode(
                        id_=hashlib.sha256(
                            raw_id.encode("utf-8")
                        ).hexdigest(),
                        text=build_image_text(
                            page_number,
                            caption,
                            position,
                            sub_caption,
                            description,
                        ),
                        embedding=embedding,
                        metadata={
                            "type": "image",
                            "page": page_number,
                            "source": pdf_path.name,
                            DOC_KEY: document_id,
                            "image_path": image_path.as_posix(),
                            "image_index": image_index,
                            "image_extension": base_image["ext"],
                            "caption": caption,
                            "position": position,
                        },
                    )

                    image_nodes.append(node)

                    logger.info(
                        "Image page %s (%s) : %s",
                        page_number,
                        position or "seule",
                        (sub_caption or caption)[:60] or "(sans légende)",
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

    # Réindexation propre : on efface les anciennes images du document
    delete_document(document_id)

    image_nodes = extract_images(
        pdf_path,
        document_id,
    )

    index_images(image_nodes)

    return {
        "document": pdf_path.name,
        DOC_KEY: document_id,
        "images": len(image_nodes),
        "with_caption": sum(
            1 for node in image_nodes
            if node.metadata["caption"]
        ),
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