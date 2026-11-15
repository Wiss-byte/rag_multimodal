"""
INGESTION DU TEXTE ET DES TABLEAUX PDF AVEC CLIP.

- Extraction PyMuPDF
- Chunking LlamaIndex compatible CLIP (<= 70 tokens)
- Extraction des tableaux (lignes reconstruites + légende)
- Embeddings CLIP 512D
- Normalisation L2
- Indexation PGVector
- Identifiant SHA-256 + réindexation propre par document
"""

import re
import sys
import hashlib
import logging
from pathlib import Path

import fitz

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.vector_stores.postgres import PGVectorStore
from transformers import CLIPTokenizerFast

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

# CLIP ne lit que 77 tokens (75 + 2 tokens spéciaux)
CHUNK_SIZE = 70
CHUNK_OVERLAP = 15
CLIP_MAX_TOKENS = 75

# Métadonnées exclues du calcul de taille des chunks
METADATA_KEYS = ["type", "page", "source", "doc_sha256"]

# Reconstruction des tableaux
TABLE_CAPTION_PATTERN = re.compile(r"^Table\s+\d+\s*:")
ROW_Y_TOLERANCE = 3.0   # mots sur la même ligne si écart vertical <= 3 pt
CELL_X_GAP = 6.0        # nouvelle cellule si écart horizontal > 6 pt


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# TOKENIZER CLIP
# ============================================================

# Même tokenizer que CLIP ViT-B-32 : on compte les tokens comme lui.
clip_tokenizer = CLIPTokenizerFast.from_pretrained(
    "openai/clip-vit-base-patch32",
    model_max_length=1_000_000,  # évite un avertissement sur les pages longues
)


def clip_tokenize(text: str) -> list[int]:
    """Tokenise un texte avec le tokenizer CLIP (sans tokens spéciaux)."""
    return clip_tokenizer.encode(text, add_special_tokens=False)


def truncate_for_clip(text: str) -> str:
    """Coupe explicitement un texte à la limite de CLIP."""

    token_ids = clip_tokenize(text)

    if len(token_ids) <= CLIP_MAX_TOKENS:
        return text

    return clip_tokenizer.decode(token_ids[:CLIP_MAX_TOKENS])


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


def delete_document(document_id: str) -> None:
    """Supprime tous les vecteurs d'un document avant réindexation."""

    text_store.delete_nodes(
        filters=MetadataFilters(
            filters=[
                MetadataFilter(
                    key="doc_sha256",
                    value=document_id,
                )
            ]
        )
    )

    logger.info(
        "Anciens vecteurs du document supprimés : %s",
        document_id[:12],
    )


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
                        "doc_sha256": document_id,
                    },
                    # Les métadonnées ne doivent pas consommer les 70 tokens
                    excluded_embed_metadata_keys=METADATA_KEYS,
                    excluded_llm_metadata_keys=METADATA_KEYS,
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
    """Découpe les pages en fragments lisibles entièrement par CLIP."""

    if not documents:
        return []

    logger.info("Découpage en chunks...")

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        tokenizer=clip_tokenize,
    )

    nodes = splitter.get_nodes_from_documents(
        documents,
        show_progress=True,
    )

    for index, node in enumerate(nodes):

        node.metadata["type"] = "text"

        document_id = node.metadata["doc_sha256"]

        page = node.metadata["page"]

        # Identifiant déterministe
        raw_id = f"{document_id}:text:{page}:{index}"

        node.id_ = hashlib.sha256(
            raw_id.encode("utf-8")
        ).hexdigest()

    # Contrôle : aucun chunk ne doit dépasser la limite CLIP
    max_tokens = max(
        len(clip_tokenize(node.get_content()))
        for node in nodes
    )

    logger.info(
        "Chunk le plus long : %s tokens CLIP (limite %s)",
        max_tokens,
        CLIP_MAX_TOKENS,
    )

    if max_tokens > CLIP_MAX_TOKENS:
        logger.warning("Certains chunks seront tronqués par CLIP.")

    logger.info("%s chunks générés.", len(nodes))

    return nodes


# ============================================================
# EXTRACTION DES TABLEAUX
# ============================================================

def find_table_caption(page, bbox) -> str:
    """Retrouve la légende « Table N: ... » placée au-dessus du tableau."""

    caption_block = None

    for block in page.get_text("blocks"):

        y1 = block[3]
        text = block[4].strip()

        if not TABLE_CAPTION_PATTERN.match(text):
            continue

        # La légende doit se trouver au-dessus du tableau
        if y1 > bbox[1] + 5:
            continue

        # On garde la légende la plus proche du tableau
        if caption_block is None or y1 > caption_block[3]:
            caption_block = block

    if caption_block is None:
        return ""

    return " ".join(caption_block[4].split())


def table_to_text(page, bbox) -> str:
    """
    Reconstruit le tableau ligne par ligne à partir de la
    position des mots (les nombres comme 28.4 restent intacts).
    """

    words = page.get_text("words", clip=fitz.Rect(bbox))

    if not words:
        return ""

    # Tri par position verticale puis horizontale
    words.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))

    # Regroupement des mots par ligne
    lines = []

    for word in words:

        y_center = (word[1] + word[3]) / 2

        if lines and abs(lines[-1]["y"] - y_center) <= ROW_Y_TOLERANCE:
            lines[-1]["words"].append(word)
        else:
            lines.append({"y": y_center, "words": [word]})

    # Découpage de chaque ligne en cellules
    rows = []

    for line in lines:

        line_words = sorted(line["words"], key=lambda w: w[0])

        cells = []
        current = []
        last_x1 = None

        for x0, _, x1, _, word_text, *_ in line_words:

            if last_x1 is not None and x0 - last_x1 > CELL_X_GAP:
                cells.append(" ".join(current))
                current = []

            current.append(word_text)
            last_x1 = x1

        cells.append(" ".join(current))

        rows.append(" | ".join(cells))

    return "\n".join(rows)


def extract_tables(
    pdf_path: Path,
    document_id: str,
) -> list[TextNode]:
    """Extrait les tableaux avec leur légende."""

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

                    bbox = tuple(table.bbox)

                    table_text = table_to_text(page, bbox)

                    if not table_text:
                        continue

                    caption = find_table_caption(page, bbox)

                    content = (
                        f"{caption}\n\n{table_text}"
                        if caption
                        else table_text
                    )

                    raw_id = (
                        f"{document_id}:table:"
                        f"{page_number}:{table_index}"
                    )

                    node = TextNode(
                        id_=hashlib.sha256(
                            raw_id.encode("utf-8")
                        ).hexdigest(),
                        text=content,
                        metadata={
                            "type": "table",
                            "page": page_number,
                            "source": pdf_path.name,
                            "doc_sha256": document_id,
                            "table_index": table_index,
                            "caption": caption,
                        },
                    )

                    table_nodes.append(node)

                    logger.info(
                        "Tableau page %s : %s",
                        page_number,
                        caption[:60] or "(sans légende)",
                    )

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

def get_embedding_text(node: TextNode) -> str:
    """
    Texte envoyé à CLIP :
    - chunk texte : le chunk lui-même ;
    - tableau : sa légende (le tableau complet dépasse 77 tokens).
    Le LLM, lui, reçoit toujours le contenu complet du node.
    """

    if node.metadata.get("type") == "table":
        text = node.metadata.get("caption") or node.get_content()
    else:
        text = node.get_content()

    return truncate_for_clip(text)


def index_nodes(nodes: list[TextNode]) -> None:
    """Calcule CLIP et stocke les nodes dans PGVector."""

    if not nodes:
        logger.warning("Aucun élément à indexer.")
        return

    logger.info(
        "Génération des embeddings CLIP : %s éléments...",
        len(nodes),
    )

    texts = [get_embedding_text(node) for node in nodes]

    embeddings = encode_texts(texts)

    for node, embedding in zip(nodes, embeddings):

        if not check_embedding(embedding):
            raise ValueError("Embedding CLIP invalide.")

        node.embedding = embedding

    # Sécurité : suppression des mêmes identifiants
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

    # Réindexation propre : on efface tout l'ancien contenu du document
    delete_document(document_id)

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
        "doc_sha256": document_id,
        "pages": len(documents),
        "chunks": len(text_nodes),
        "tables": len(table_nodes),
        "chunk_size_clip_tokens": CHUNK_SIZE,
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