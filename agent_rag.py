"""
RECHERCHE MULTIMODALE + OUTIL AGENTIQUE (retriever_tool).

Recherche (étape 4), par similarité cosinus :
1. Texte -> Texte / Tableaux
2. Texte -> Image
3. Image -> Texte
4. Image -> Image

Outil pour l'agent (étape 5) : retriever_tool

Embeddings : CLIP ViT-B-32 (512D, normalisés L2)
Base vectorielle : PostgreSQL / PGVector
"""

import os
import re
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from langchain_core.tools import tool

from llama_index.core.vector_stores import (
    FilterOperator,
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

DOC_KEY = "doc_sha256"

MODALITIES = ("all", "text", "table", "image")

DEFAULT_K = 5        # nombre de résultats texte / tableaux
MIN_K = 3            # garde-fou : un petit LLM demande parfois k=1
IMAGE_K_IN_ALL = 1   # images ajoutées en mode "all"
MAX_K = 10

# Références explicites dans la requête : « Figure 2 », « Table 3 », « tableau 1 »
REFERENCE_PATTERN = re.compile(
    r"\b(fig(?:ure)?|table|tableau)\.?\s*(\d+)",
    re.IGNORECASE,
)

TYPE_LABELS = {
    "text": "texte",
    "table": "tableau",
    "image": "image",
}


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
    k: int,
    doc_type: str | None = None,
    extra_filters: list[MetadataFilter] | None = None,
) -> list[dict]:
    """
    Recherche par similarité cosinus via PGVector.
    Score = 1 - distance cosinus (1 = identique).
    """

    filter_list = list(extra_filters or [])

    if doc_type:
        filter_list.append(
            MetadataFilter(
                key="type",
                value=doc_type,
            )
        )

    filters = MetadataFilters(filters=filter_list) if filter_list else None

    result = store.query(
        VectorStoreQuery(
            query_embedding=vector,
            similarity_top_k=k,
            filters=filters,
        )
    )

    similarities = result.similarities or []

    results = []

    for index, node in enumerate(result.nodes or []):

        metadata = node.metadata

        score = (
            round(float(similarities[index]), 4)
            if index < len(similarities)
            else None
        )

        results.append({
            "node_id": node.node_id,
            "type": metadata.get("type", "text"),
            "page": metadata.get("page"),
            "score": score,
            "content": node.get_content(),
            "image_path": metadata.get("image_path"),
            "caption": metadata.get("caption", ""),
            DOC_KEY: metadata.get(DOC_KEY),
            "explicit": False,
        })

    return results


def encode_query(
    query: str | None = None,
    image_path: str | None = None,
) -> list[float]:
    """Encode la requête (image prioritaire si fournie) dans l'espace CLIP."""

    if image_path:

        path = Path(image_path)

        if not path.is_file():
            raise FileNotFoundError(f"Image introuvable : {path}")

        return encode_image(path)

    if not query or not query.strip():
        raise ValueError("La requête est vide.")

    return encode_text(query)


# ============================================================
# REFERENCES EXPLICITES (« Figure 2 », « Table 3 »)
# ============================================================

def find_references(query: str | None) -> list[tuple[str, int]]:
    """Repère les figures et tableaux cités dans la requête."""

    references = []

    for word, number in REFERENCE_PATTERN.findall(query or ""):

        kind = "table" if word.lower().startswith("tab") else "image"

        references.append((kind, int(number)))

    return list(dict.fromkeys(references))


def lookup_references(
    query: str | None,
    vector: list[float],
) -> list[dict]:
    """
    Recherche hybride : en plus de la similarité vectorielle, on va
    chercher directement par métadonnée la figure ou le tableau cité
    (légende commençant par « Figure N: » ou « Table N: »).
    """

    results = []

    for kind, number in find_references(query):

        label = f"{'Table' if kind == 'table' else 'Figure'} {number}:"

        store = store_text if kind == "table" else store_images

        matches = vector_search(
            store,
            vector,
            k=4,
            doc_type=kind,
            extra_filters=[
                MetadataFilter(
                    key="caption",
                    value=label,
                    operator=FilterOperator.TEXT_MATCH,
                )
            ],
        )

        for match in matches:
            match["explicit"] = True

        results.extend(matches)

    return results


def deduplicate(results: list[dict]) -> list[dict]:
    """Supprime les doublons en gardant le premier (le mieux placé)."""

    seen = set()
    unique = []

    for result in results:

        if result["node_id"] in seen:
            continue

        seen.add(result["node_id"])
        unique.append(result)

    return unique


# ============================================================
# RECHERCHE MULTIMODALE
# ============================================================

def retrieve(
    query: str | None = None,
    image_path: str | None = None,
    modality: str = "all",
    k: int = DEFAULT_K,
) -> list[dict]:
    """
    Recherche multimodale : la requête est un texte OU une image.

    modality :
    - "text"  : chunks textuels
    - "table" : tableaux
    - "image" : figures
    - "all"   : texte + tableaux, puis la meilleure figure

    Si la requête texte cite « Figure N » ou « Table N », l'élément cité
    est toujours ajouté en tête, quelle que soit la modalité choisie.
    """

    if modality not in MODALITIES:
        raise ValueError(
            f"Modalité inconnue : {modality}. "
            f"Valeurs possibles : {', '.join(MODALITIES)}"
        )

    k = max(MIN_K, min(int(k), MAX_K))

    vector = encode_query(query, image_path)

    if modality == "text":
        results = vector_search(store_text, vector, k, "text")

    elif modality == "table":
        results = vector_search(store_text, vector, k, "table")

    elif modality == "image":
        results = vector_search(store_images, vector, k, "image")

    else:
        # Mode "all" : les scores texte et image ne sont pas comparables
        # (écart de modalité de CLIP), on ne les mélange donc pas dans
        # un même classement : texte/tableaux d'abord, puis la meilleure image.
        results = (
            vector_search(store_text, vector, k)
            + vector_search(store_images, vector, IMAGE_K_IN_ALL, "image")
        )

    # Recherche hybride : références explicites en tête (requête texte seulement)
    if not image_path:
        results = lookup_references(query, vector) + results

    return deduplicate(results)


# ============================================================
# FONCTIONS PAR DIRECTION (utilisées par app.py et les tests)
# ============================================================

def retrieve_text(query: str, k: int = 4) -> list[dict]:
    """Texte -> Texte."""
    return retrieve(query=query, modality="text", k=k)


def retrieve_images_from_text(query: str, k: int = 3) -> list[dict]:
    """Texte -> Image."""
    return retrieve(query=query, modality="image", k=k)


def retrieve_text_from_image(image_path: str, k: int = 4) -> list[dict]:
    """Image -> Texte."""
    return retrieve(image_path=image_path, modality="text", k=k)


def search_similar_images(image_path: str, k: int = 3) -> list[dict]:
    """Image -> Image."""
    return retrieve(image_path=image_path, modality="image", k=k)


# ============================================================
# FORMATAGE DES RESULTATS
# ============================================================

def format_results(results: list[dict]) -> str:
    """Formate les résultats pour le LLM (et pour l'affichage console)."""

    if not results:
        return "Aucun résultat trouvé dans le document."

    blocks = []

    for rank, result in enumerate(results, start=1):

        score = (
            f"{result['score']:.3f}"
            if result["score"] is not None
            else "?"
        )

        origin = " | référence citée" if result.get("explicit") else ""

        if result["type"] == "image":

            blocks.append(
                f"[{rank}] IMAGE: {result['image_path']} | "
                f"Page: {result['page']} | Score: {score}{origin}\n"
                f"{result['content']}"
            )

        else:

            label = TYPE_LABELS.get(result["type"], result["type"])

            blocks.append(
                f"[{rank}] ({label}, page {result['page']}, "
                f"score {score}{origin})\n"
                f"{result['content']}"
            )

    return "\n\n".join(blocks)


# ============================================================
# OUTIL AGENTIQUE : retriever_tool
# ============================================================

TOOL_DESCRIPTION = """Recherche des passages, tableaux et figures dans l'article
« Attention Is All You Need » (recherche vectorielle CLIP, similarité cosinus).

Quand l'utiliser : avant toute réponse sur le contenu de l'article.

Choix de modality :
- "text"  : définitions, explications, méthodes ;
- "table" : scores, chiffres, hyperparamètres, comparaisons (BLEU, coûts...) ;
- "image" : figures, schémas, architecture visuelle ;
- "all"   : question qui mélange texte et figure (valeur par défaut).

Règles :
- écris query en anglais (le document et CLIP sont en anglais) ;
- si la question cite une figure ou un tableau, garde « Figure N » ou
  « Table N » dans query : l'élément cité sera ajouté automatiquement ;
- si l'utilisateur fournit le chemin d'une image, mets-le dans image_path ;
- chaque résultat indique son type, sa page et son score de similarité ;
  les figures apparaissent sous la forme « IMAGE: chemin | Page: n »."""


class RetrieverInput(BaseModel):
    """Arguments de retriever_tool."""

    query: str = Field(
        description=(
            "Requête de recherche en anglais, "
            "ex. 'multi-head attention definition'."
        )
    )

    modality: Literal["all", "text", "table", "image"] = Field(
        default="all",
        description="Type de contenu recherché : all, text, table ou image.",
    )

    image_path: Optional[str] = Field(
        default=None,
        description=(
            "Chemin d'une image fournie par l'utilisateur "
            "(recherche par image). Laisser vide sinon."
        ),
    )

    k: int = Field(
        default=DEFAULT_K,
        ge=1,
        le=MAX_K,
        description="Nombre de résultats à retourner (3 minimum).",
    )


@tool(
    "retriever_tool",
    args_schema=RetrieverInput,
    description=TOOL_DESCRIPTION,
)
def retriever_tool(
    query: str,
    modality: str = "all",
    image_path: Optional[str] = None,
    k: int = DEFAULT_K,
) -> str:

    # Les petits LLM envoient parfois "None" ou "" au lieu de rien
    if image_path and image_path.strip().lower() in {"", "none", "null"}:
        image_path = None

    try:
        results = retrieve(
            query=query,
            image_path=image_path,
            modality=modality,
            k=k,
        )

    except (FileNotFoundError, ValueError) as error:
        return f"Erreur de recherche : {error}"

    return format_results(results)


# ============================================================
# OUTILS DE L'AGENT
# ============================================================

TOOLS = [retriever_tool]


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


# ============================================================
# OUVERTURE D'UNE IMAGE (option --show)
# ============================================================

def open_image(image_path: str) -> None:
    """Ouvre l'image avec la visionneuse du système, sans fichier temporaire."""

    if sys.platform.startswith("win"):
        os.startfile(image_path)
    elif sys.platform == "darwin":
        subprocess.run(["open", image_path], check=False)
    else:
        subprocess.run(["xdg-open", image_path], check=False)


# ============================================================
# EXECUTION : recherche directe (sans agent)
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Recherche multimodale top-k (sans agent)."
    )

    parser.add_argument("query", nargs="?", default="")
    parser.add_argument("--image", default=None, help="Image requête")
    parser.add_argument("--modality", default="all", choices=MODALITIES)
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Ouvre les images trouvées",
    )

    args = parser.parse_args()

    found = retrieve(
        query=args.query,
        image_path=args.image,
        modality=args.modality,
        k=args.k,
    )

    print(format_results(found))

    if args.show:
        for item in found:
            if item["type"] == "image" and item["image_path"]:
                open_image(item["image_path"])