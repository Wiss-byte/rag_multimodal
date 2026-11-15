
"""
Utilitaires de gestion des documents.
"""

import hashlib
from pathlib import Path


def calculate_document_id(file_path: str) -> str:
    """
    Calcule l'identifiant SHA-256 d'un document PDF.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            sha256.update(block)

    return sha256.hexdigest()


if __name__ == "__main__":

    path = "data/attention.pdf"

    document_id = calculate_document_id(path)

    print("Document ID :", document_id)
