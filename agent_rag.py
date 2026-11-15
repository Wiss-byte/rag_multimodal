
"""
RAG multimodal :
- Recherche texte vers texte
- Recherche texte vers image
- Recherche image vers image
- Outils pour l'agent LangGraph
"""

import re
from functools import lru_cache
from pathlib import Path

from PIL import Image
from sentence_transformers import SentenceTransformer

from langchain.tools import tool

from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore


# ============================================================
# CONFIGURATION
# ============================================================

DB = {
    "host": "localhost",
    "port": 5433,
    "database": "rag",
    "user": "rag",
    "password": "rag",
}

# Modèle textuel : 384 dimensions
embed = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Modèle multimodal : 512 dimensions
@lru_cache(maxsize=1)
def clip_model():
    return SentenceTransformer("clip-ViT-B-32")


# Bases vectorielles
store_text = PGVectorStore.from_params(
    table_name="text_chunks",
    embed_dim=384,
    **DB,
)

store_caps = PGVectorStore.from_params(
    table_name="image_captions",
    embed_dim=384,
    **DB,
)

store_clip = PGVectorStore.from_params(
    table_name="image_clip",
    embed_dim=512,
    **DB,
)


# ============================================================
# RECHERCHE VECTORIELLE GENERIQUE
# ============================================================

def vector_search(store, vector, k=3, doc_type=None):

    filters = None

    if doc_type:
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="type",
                    value=doc_type
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
# 1. TEXTE VERS TEXTE
# ============================================================

def retrieve_text(query: str, k: int = 4):

    vector = embed.get_query_embedding(query)

    result = vector_search(
        store_text,
        vector,
        k,
        "text"
    )

    return result.nodes


# ============================================================
# 2. TEXTE VERS IMAGE
# ============================================================

def retrieve_images_from_text(query: str, k: int = 3):

    # Embedding CLIP du texte
    vector = clip_model().encode(
        query,
        normalize_embeddings=True
    ).tolist()

    result = vector_search(
        store_clip,
        vector,
        k
    )

    return result.nodes


# ============================================================
# 3. IMAGE VERS IMAGE
# ============================================================

def search_similar_images(image_path: str, k: int = 3):

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image introuvable : {image_path}"
        )

    # Charger l'image
    with Image.open(path) as image:
        image = image.convert("RGB")

        # Générer l'embedding CLIP normalisé
        vector = clip_model().encode(
            image,
            normalize_embeddings=True
        ).tolist()

    # Recherche dans PGVector
    result = vector_search(
        store_clip,
        vector,
        k
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
            "image_path": node.metadata["image_path"],
            "page": node.metadata["page"],
            "score": score,
        })

    return results


# ============================================================
# OUTILS DE L'AGENT
# ============================================================

@tool
def search_text(query: str) -> str:
    """Recherche les passages textuels du PDF."""

    nodes = retrieve_text(query)

    return "\n\n".join(
        f"[Page {node.metadata['page']}] "
        f"{node.get_content()}"
        for node in nodes
    ) or "Aucun résultat textuel."


@tool
def search_tables(query: str) -> str:
    """Recherche les tableaux et résultats numériques du PDF."""

    vector = embed.get_query_embedding(query)

    result = vector_search(
        store_text,
        vector,
        3,
        "table"
    )

    return "\n\n".join(
        f"[Page {node.metadata['page']}]\n"
        f"{node.get_content()}"
        for node in result.nodes
    ) or "Aucun tableau trouvé."


@tool
def search_images(query: str) -> str:
    """Recherche les figures et illustrations du PDF à partir d'une description textuelle."""

    found = {}

    # Recherche par descriptions
    vector = embed.get_query_embedding(query)

    result = vector_search(
        store_caps,
        vector,
        3
    )

    for node in result.nodes:

        path = node.metadata["image_path"]

        found[path] = (
            node.metadata["page"],
            node.get_content()[:300]
        )

    # Recherche avec CLIP
    nodes = retrieve_images_from_text(query, 3)

    for node in nodes:

        path = node.metadata["image_path"]

        found.setdefault(
            path,
            (
                node.metadata["page"],
                "Résultat trouvé par CLIP"
            )
        )

    return "\n".join(
        f"IMAGE: {path} | Page: {page} | {description}"
        for path, (page, description) in found.items()
    ) or "Aucune image trouvée."


TOOLS = [
    search_text,
    search_tables,
    search_images,
]


# ============================================================
# EXTRACTION DES IMAGES D'UNE REPONSE
# ============================================================

def extract_images(outputs):

    text = "\n".join(str(output) for output in outputs)

    paths = re.findall(
        r"IMAGE:\s*(\S+)",
        text
    )

    return list(dict.fromkeys(paths))
