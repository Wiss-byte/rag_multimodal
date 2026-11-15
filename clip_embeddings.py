
"""
Embeddings multimodaux avec CLIP ViT-B-32.
Texte et images partagent un espace latent de 512 dimensions.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer


MODEL_NAME = "clip-ViT-B-32"
EMBEDDING_DIM = 512


@lru_cache(maxsize=1)
def get_clip_model():
    """Charge CLIP une seule fois."""
    print("Chargement de CLIP ViT-B-32...")
    model = SentenceTransformer(MODEL_NAME)
    print("CLIP chargé.")
    return model


def encode_text(text: str) -> list[float]:
    """Encode un texte dans l'espace latent CLIP."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Le texte ne peut pas être vide.")

    vector = get_clip_model().encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return np.asarray(vector, dtype=np.float32).tolist()


def encode_texts(texts: list[str]) -> list[list[float]]:
    """Encode plusieurs textes par batch."""

    if not texts:
        return []

    if any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Tous les textes doivent être non vides.")

    vectors = get_clip_model().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )

    return np.asarray(vectors, dtype=np.float32).tolist()


def encode_image_object(image: Image.Image) -> list[float]:
    """Encode une image PIL."""

    if not isinstance(image, Image.Image):
        raise TypeError("L'entrée doit être une image PIL.")

    image = image.convert("RGB")

    vector = get_clip_model().encode(
        image,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return np.asarray(vector, dtype=np.float32).tolist()


def encode_image(image_path: str | Path) -> list[float]:
    """Encode une image à partir de son chemin."""

    path = Path(image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image introuvable : {path}")

    with Image.open(path) as image:
        return encode_image_object(image)


def check_embedding(vector: list[float]) -> bool:
    """Vérifie la dimension et la normalisation L2."""

    array = np.asarray(vector, dtype=np.float32)

    return (
        array.shape == (EMBEDDING_DIM,)
        and np.isfinite(array).all()
        and np.isclose(
            np.linalg.norm(array),
            1.0,
            atol=1e-4,
        )
    )
