
"""
AGENT RAG MULTIMODAL.

Recherche :
1. Texte -> Texte
2. Texte -> Image
3. Image -> Texte
4. Image -> Image

Outils :
- search_text
- search_tables
- search_images

Embeddings : CLIP ViT-B-32 (512D)
Base vectorielle : PostgreSQL / PGVector
"""

import re
from pathlib import Path

from PIL import Image

from langchain.tools import tool

from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from llama_index.vector_stores.postgres import PGVectorStore

from clip_embeddings import (
    encode_text,
    encode_image,
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
IMAGE_TABLE = "image_clip"


# ============================================================
# BASES VECTORIELLES
# ============================================================

store_text = PGVectorStore.from_params(
    table_name=TEXT_TABLE,
    embed_dim=EMBEDDING_DIM,
    **DB_CONFIG,
)

store_images = PGVectorStore.from_params(
    table_name=IMAGE_TABLE,
    embed_dim=EMBEDDING_DIM,
    **DB_CONFIG,
)


# ============================================================
# RECHERCHE VECTORIELLE GENERIQUE
# ============================================================

def vector_search(
    store,
    vector: list[float],
    k: int = 3,
    doc_type: str | None = None,
):
    """Recherche par similarité cosinus via PGVector."""

    filters = None

    if doc_type:

        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="type",
                    value=doc_type,
                )
            ]
        )

    query = VectorStoreQuery(
        query_embedding=vector,
        similarity_top_k=k,
        filters=filters,
    )

    return store.query(query)


# ============================================================
# 1. TEXTE -> TEXTE
# ============================================================

def retrieve_text(
    query: str,
    k: int = 4,
):
    """Recherche des chunks textuels à partir d'une question."""

    vector = encode_text(query)

    result = vector_search(
        store_text,
        vector,
        k,
        doc_type="text",
    )

    return result.nodes


# ============================================================
# 2. TEXTE -> IMAGE
# ============================================================

def retrieve_images_from_text(
    query: str,
    k: int = 3,
):
    """Recherche les images à partir d'une description textuelle."""

    vector = encode_text(query)

    result = vector_search(
        store_images,
        vector,
        k,
        doc_type="image",
    )

    return result.nodes


# ============================================================
# 3. IMAGE -> TEXTE
# ============================================================

def retrieve_text_from_image(
    image_path: str,
    k: int = 4,
):
    """Recherche les passages textuels associés à une image."""

    vector = encode_image(image_path)

    result = vector_search(
        store_text,
        vector,
        k,
    )

    return result.nodes


# ============================================================
# 4. IMAGE -> IMAGE
# ============================================================

def search_similar_images(
    image_path: str,
    k: int = 3,
):
    """Recherche les images visuellement similaires."""

    vector = encode_image(image_path)

    result = vector_search(
        store_images,
        vector,
        k,
        doc_type="image",
    )

    results = []

    similarities = result.similarities or []

    for index, node in enumerate(result.nodes):

        score = (
            float(similarities[index])
            if index < len(similarities)
            else None
        )

        results.append({
            "image_path": node.metadata.get("image_path"),
            "page": node.metadata.get("page"),
            "document_id": node.metadata.get("document_id"),
            "score": score,
        })

    return results


# ============================================================
# FORMATAGE DES RESULTATS
# ============================================================

def format_text_results(nodes) -> str:

    if not nodes:
        return "Aucun résultat textuel trouvé."

    return "\n\n".join(
        (
            f"[Page {node.metadata.get('page', '?')}] "
            f"{node.get_content()}"
        )
        for node in nodes
    )


def format_image_results(nodes) -> str:

    if not nodes:
        return "Aucune image trouvée."

    results = []

    for node in nodes:

        path = node.metadata.get("image_path", "")

        page = node.metadata.get("page", "?")

        results.append(
            f"IMAGE: {path} | Page: {page}"
        )

    return "\n".join(results)


# ============================================================
# OUTIL 1 : RECHERCHE TEXTUELLE
# ============================================================

@tool
def search_text(query: str) -> str:
    """
    Recherche les passages textuels du PDF.

    Utiliser pour les définitions, concepts,
    explications et descriptions de méthodes.
    """

    nodes = retrieve_text(query, k=4)

    return format_text_results(nodes)


# ============================================================
# OUTIL 2 : RECHERCHE DE TABLEAUX
# ============================================================

@tool
def search_tables(query: str) -> str:
    """
    Recherche les tableaux du PDF.

    Utiliser pour les scores, résultats numériques,
    comparaisons et données expérimentales.
    """

    vector = encode_text(query)

    result = vector_search(
        store_text,
        vector,
        k=4,
        doc_type="table",
    )

    return format_text_results(result.nodes)


# ============================================================
# OUTIL 3 : RECHERCHE D'IMAGES
# ============================================================

@tool
def search_images(query: str) -> str:
    """
    Recherche les figures et les schémas du PDF
    à partir d'une description textuelle.

    Utiliser pour les architectures, diagrammes,
    illustrations et représentations visuelles.
    """

    nodes = retrieve_images_from_text(query, k=4)

    return format_image_results(nodes)


# ============================================================
# OUTILS DE L'AGENT
# ============================================================

TOOLS = [
    search_text,
    search_tables,
    search_images,
]


# ============================================================
# EXTRACTION DES IMAGES D'UNE REPONSE
# ============================================================

def extract_images(outputs) -> list[str]:
    """Récupère les chemins d'images dans les observations."""

    text = "\n".join(
        str(output)
        for output in outputs
    )

    paths = re.findall(
        r"IMAGE:\s*(.+?)\s*\|\s*Page:",
        text,
    )

    return list(dict.fromkeys(paths))
