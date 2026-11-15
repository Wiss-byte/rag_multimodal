
import numpy as np
import pytest

from PIL import Image

import clip_embeddings as clip


class FakeCLIP:

    def encode(
        self,
        inputs,
        convert_to_numpy=True,
        normalize_embeddings=True,
        **kwargs,
    ):

        if isinstance(inputs, list):

            vectors = np.ones(
                (len(inputs), clip.EMBEDDING_DIM),
                dtype=np.float32,
            )

        else:

            vectors = np.ones(
                clip.EMBEDDING_DIM,
                dtype=np.float32,
            )

        if normalize_embeddings:

            if vectors.ndim == 1:
                vectors /= np.linalg.norm(vectors)
            else:
                vectors /= np.linalg.norm(
                    vectors,
                    axis=1,
                    keepdims=True,
                )

        return vectors


@pytest.fixture
def fake_model(monkeypatch):

    monkeypatch.setattr(
        clip,
        "get_clip_model",
        lambda: FakeCLIP(),
    )


def test_encode_text(fake_model):

    vector = clip.encode_text(
        "Transformer architecture"
    )

    assert len(vector) == 512
    assert clip.check_embedding(vector)


def test_encode_texts(fake_model):

    vectors = clip.encode_texts([
        "Self attention",
        "Multi head attention",
    ])

    assert len(vectors) == 2

    for vector in vectors:
        assert clip.check_embedding(vector)


def test_encode_empty_list():

    assert clip.encode_texts([]) == []


def test_empty_text():

    with pytest.raises(ValueError):
        clip.encode_text("")


def test_encode_image(fake_model, tmp_path):

    image_path = tmp_path / "test.png"

    image = Image.new(
        "RGB",
        (100, 100),
        color="white",
    )

    image.save(image_path)

    vector = clip.encode_image(image_path)

    assert len(vector) == 512
    assert clip.check_embedding(vector)


def test_missing_image():

    with pytest.raises(FileNotFoundError):
        clip.encode_image("missing_image.png")


def test_invalid_embedding():

    vector = [0.5] * 512

    assert not clip.check_embedding(vector)


def test_embedding_dimension():

    assert clip.EMBEDDING_DIM == 512
