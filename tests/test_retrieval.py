
from pathlib import Path

from agent_rag import (
    retrieve_text,
    retrieve_images_from_text,
    search_similar_images,
)


def test_text_retrieval():

    results = retrieve_text(
        "What is the attention mechanism?",
        k=3
    )

    assert len(results) > 0


def test_text_to_image():

    results = retrieve_images_from_text(
        "Transformer encoder decoder architecture",
        k=3
    )

    assert len(results) > 0


def test_image_to_image():

    images = list(Path("data/images").glob("*.png"))

    assert images, "Aucune image indexée."

    results = search_similar_images(
        str(images[0]),
        k=3
    )

    assert len(results) > 0

    assert "image_path" in results[0]
    assert "page" in results[0]


def test_image_metadata():

    images = list(Path("data/images").glob("*.png"))

    assert images

    results = search_similar_images(
        str(images[0]),
        k=3
    )

    for result in results:

        assert result["page"] > 0
        assert Path(result["image_path"]).exists()
