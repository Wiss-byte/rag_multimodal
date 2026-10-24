"""Indexation des figures : embeddings CLIP + descriptions (legende du PDF + moondream) -> PGVector.
Usage : uv run python ingest_images.py [chemin_du_pdf]"""
import re
import sys
from pathlib import Path

import ollama
import pymupdf
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from PIL import Image
from sentence_transformers import SentenceTransformer

PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/attention.pdf")
IMG_DIR = Path("data/images")
SOURCE = PDF.name
VLM = "moondream"
DB = dict(host="localhost", port=5433, database="rag", user="rag", password="rag")

# ---------------------------------------------------------------- modeles
clip = SentenceTransformer("clip-ViT-B-32")                                             # 512 dim
text_embed = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")  # 384 dim

# Deux tables : les dimensions sont differentes (512 pour CLIP, 384 pour le texte)
store_clip = PGVectorStore.from_params(table_name="image_clip", embed_dim=512, **DB)
store_caps = PGVectorStore.from_params(table_name="image_captions", embed_dim=384, **DB)
store_clip.clear()
store_caps.clear()

pdf = pymupdf.open(PDF)

# ---------------------------------------------------------------- fonctions utiles
# Plusieurs essais : un petit modele renvoie parfois une reponse vide ou en boucle
ATTEMPTS = [
    ("Describe this figure in 3 or 4 sentences: main components, labels, and what it shows.",
     {"temperature": 0, "num_predict": 150, "repeat_penalty": 1.3}),
    ("Describe this figure or diagram in detail.",
     {"temperature": 0.3, "num_predict": 150, "repeat_penalty": 1.1}),
    ("What is in this image?",
     {"temperature": 0.5, "num_predict": 120}),
]


def describe(path: Path) -> str:
    """Description par le modele vision-langage ; essaie plusieurs prompts si la reponse est vide."""
    for prompt, opts in ATTEMPTS:
        r = ollama.chat(
            model=VLM,
            messages=[{"role": "user", "content": prompt, "images": [str(path)]}],
            options=opts,
        )
        text = r["message"]["content"].strip()
        if len(text) > 30:
            return text
    return ""


def pdf_caption(page_number: int) -> str:
    """Legende(s) 'Figure N: ...' extraite(s) du texte de la page du PDF."""
    text = pdf[page_number - 1].get_text()
    found = re.findall(r"Figure\s+\d+\s*:.{0,200}", text, flags=re.S)
    return " ".join(" ".join(f.split()) for f in found[:2])


# ---------------------------------------------------------------- indexation
clip_nodes, cap_nodes = [], []
for path in sorted(IMG_DIR.glob("*.png")):
    page = int(path.stem.split("_")[0][1:])          # "p3_1" -> 3
    meta = {"type": "image", "page": page, "source": SOURCE, "image_path": path.as_posix()}

    # 1) CLIP : vecteur de l'image (permet de la chercher avec une phrase)
    with Image.open(path) as im:
        v = clip.encode(im.convert("RGB"), normalize_embeddings=True).tolist()
    clip_nodes.append(TextNode(text=path.as_posix(), embedding=v, metadata=meta))

    # 2) Description = legende du PDF + description du modele vision-langage
    parts = [pdf_caption(page), describe(path)]
    caption = " ".join(p for p in parts if p) or f"Figure on page {page} of the document."
    print(f"\n--- {path.name} (page {page}) ---\n{caption}")

    # 3) La description est embeddee comme du texte -> interrogeable
    cap_nodes.append(TextNode(
        text=caption, embedding=text_embed.get_text_embedding(caption), metadata=meta))

if clip_nodes:
    store_clip.add(clip_nodes)
    store_caps.add(cap_nodes)
print(f"\nIndexe : {len(clip_nodes)} images (CLIP) et {len(cap_nodes)} descriptions.")

# ---------------------------------------------------------------- tests de recherche (uniquement en mode console)
if len(sys.argv) == 1 and clip_nodes:
    for q in ["Transformer model architecture with encoder and decoder",
              "scaled dot-product attention and multi-head attention"]:
        print(f"\nRequete : {q}")
        qv = clip.encode(q, normalize_embeddings=True).tolist()
        res = store_clip.query(VectorStoreQuery(query_embedding=qv, similarity_top_k=2))
        for n, s in zip(res.nodes, res.similarities):
            print(f"  CLIP        -> {n.metadata['image_path']} (page {n.metadata['page']}) score={s:.2f}")
        qt = text_embed.get_query_embedding(q)
        res = store_caps.query(VectorStoreQuery(query_embedding=qt, similarity_top_k=2))
        for n, s in zip(res.nodes, res.similarities):
            print(f"  Description -> {n.metadata['image_path']} (page {n.metadata['page']}) score={s:.2f}")