"""Extraction (texte, tableaux, figures) d'un PDF avec LlamaIndex + embeddings open source -> PGVector.
Usage : uv run python ingest.py [chemin_du_pdf]"""
import sys
from pathlib import Path

import pymupdf
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PyMuPDFReader
from llama_index.vector_stores.postgres import PGVectorStore

PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/attention.pdf")
IMG_DIR = Path("data/images")
IMG_DIR.mkdir(parents=True, exist_ok=True)
for old in IMG_DIR.glob("*.png"):          # efface les figures d'un PDF precedent
    old.unlink()

# ---------------------------------------------------------------- 1. TEXTE (LlamaIndex)
# Un Document par page ; metadonnees courtes (type, page, source)
docs = PyMuPDFReader().load_data(file_path=PDF)
for d in docs:
    page = int(d.metadata["source"])
    d.metadata = {"type": "text", "page": page, "source": PDF.name}

# MiniLM tronque a ~256 tokens -> chunks courts
splitter = SentenceSplitter(chunk_size=200, chunk_overlap=30)
text_nodes = splitter.get_nodes_from_documents(docs)
print(f"Texte : {len(docs)} pages -> {len(text_nodes)} chunks")


# ---------------------------------------------------------------- 2. TABLEAUX (PyMuPDF -> Markdown)
def to_markdown(rows):
    """Convertit un tableau (liste de lignes) en tableau Markdown."""
    rows = [[(c or "").replace("\n", " ").strip() for c in r] for r in rows if r]
    if len(rows) < 2:
        return ""
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    md = "| " + " | ".join(rows[0]) + " |\n|" + "---|" * w + "\n"
    return md + "\n".join("| " + " | ".join(r) + " |" for r in rows[1:])


table_nodes, table_boxes = [], {}      # table_boxes[page] = [bbox...] (pour exclure des figures)
doc_tables = pymupdf.open(PDF)
for i, page in enumerate(doc_tables, 1):
    for t in page.find_tables(strategy="lines").tables:
        md = to_markdown(t.extract())
        if md:
            table_boxes.setdefault(i, []).append(tuple(t.bbox))
            table_nodes.append(TextNode(
                text=md, metadata={"type": "table", "page": i, "source": PDF.name}))
print(f"Tableaux : {len(table_nodes)}")
for n in table_nodes:
    print(f"--- page {n.metadata['page']} ---\n{n.text[:300]}\n")


# ---------------------------------------------------------------- 3. FIGURES (PyMuPDF)
# Les figures de l'article sont vectorielles : on regroupe les tracés et on rend la zone en PNG
def inside(r, b):
    """Le centre du rectangle r est-il dans la boite b ?"""
    cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
    return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]


images = []   # (page, chemin)
doc = pymupdf.open(PDF)
for i, page in enumerate(doc, 1):
    rects = [pymupdf.Rect(r) for r in page.cluster_drawings()]
    for xref in {im[0] for im in page.get_images(full=True)}:
        rects += [pymupdf.Rect(r) for r in page.get_image_rects(xref)]
    kept = []
    for r in sorted(rects, key=lambda r: -r.get_area()):
        if r.width < 100 or r.height < 80:
            continue                                            # icones, filets
        if any(inside(r, b) for b in table_boxes.get(i, [])):
            continue                                            # c'est un tableau
        if any(k.contains(r) for k in kept):
            continue                                            # doublon
        kept.append(r)
    for k, r in enumerate(kept, 1):
        clip = pymupdf.Rect(r.x0 - 4, r.y0 - 4, r.x1 + 4, r.y1 + 4) & page.rect
        out = IMG_DIR / f"p{i}_{k}.png"
        page.get_pixmap(clip=clip, dpi=110).save(out)
        images.append((i, out.as_posix()))
print(f"Images : {len(images)} -> {images}")


# ---------------------------------------------------------------- 4. EMBEDDINGS + PGVECTOR
embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")  # 384 dim

vector_store = PGVectorStore.from_params(
    host="localhost", port=5433, database="rag", user="rag", password="rag",
    table_name="text_chunks",          # table reelle : data_text_chunks
    embed_dim=384,
)
vector_store.clear()                    # evite les doublons si on relance

storage_context = StorageContext.from_defaults(vector_store=vector_store)
VectorStoreIndex(
    text_nodes + table_nodes,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True,
)
print("Indexation terminee.")