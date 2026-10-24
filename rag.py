"""RAG agentique : un agent LangChain (Qwen/Ollama) choisit parmi 3 outils de recherche PGVector.
Usage : uv run python agent_rag.py "votre question" """
import re
import sys
from functools import lru_cache

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_ollama import ChatOllama
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, VectorStoreQuery
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from sentence_transformers import SentenceTransformer

LLM = "qwen3:1.7b"   # alternative plus fiable : "qwen3:4b" (ollama pull qwen3:4b)
DB = dict(host="localhost", port=5433, database="rag", user="rag", password="rag")

embed = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
store_text = PGVectorStore.from_params(table_name="text_chunks", embed_dim=384, **DB)
store_caps = PGVectorStore.from_params(table_name="image_captions", embed_dim=384, **DB)
store_clip = PGVectorStore.from_params(table_name="image_clip", embed_dim=512, **DB)


@lru_cache(maxsize=1)
def clip_model():
    return SentenceTransformer("clip-ViT-B-32")


def _search(store, vec, k, doc_type=None):
    """Recherche vectorielle, avec filtre optionnel sur la metadonnee 'type' (text / table)."""
    filters = None
    if doc_type:
        filters = MetadataFilters(filters=[MetadataFilter(key="type", value=doc_type)])
    q = VectorStoreQuery(query_embedding=vec, similarity_top_k=k, filters=filters)
    return store.query(q).nodes


# ---------------------------------------------------------------- les 3 outils de l'agent
@tool
def search_text(query: str) -> str:
    """Search text passages of the PDF: concepts, explanations, definitions, methods."""
    nodes = _search(store_text, embed.get_query_embedding(query), 4, "text")
    return "\n\n".join(f"[page {n.metadata['page']}] {n.get_content()}" for n in nodes) or "No result."


@tool
def search_tables(query: str) -> str:
    """Search tables of the PDF: numeric results, BLEU scores, costs, hyperparameters."""
    nodes = _search(store_text, embed.get_query_embedding(query), 2, "table")
    return "\n\n".join(f"[page {n.metadata['page']}] TABLE:\n{n.get_content()}" for n in nodes) or "No table."


@tool
def search_images(query: str) -> str:
    """Search figures and diagrams of the PDF: architecture, schemas, illustrations."""
    found = {}
    # 1) par description textuelle
    for n in _search(store_caps, embed.get_query_embedding(query), 1):
        found[n.metadata["image_path"]] = (n.metadata["page"], n.get_content()[:300])
    # 2) par CLIP (texte -> image)
    qv = clip_model().encode(query, normalize_embeddings=True).tolist()
    for n in _search(store_clip, qv, 1):
        found.setdefault(n.metadata["image_path"], (n.metadata["page"], "(found by CLIP)"))
    return "\n".join(f"IMAGE: {p} | page {pg} | description: {d}" for p, (pg, d) in found.items()) or "No image."


SYSTEM = """Tu es un assistant qui repond aux questions sur un document PDF.
Regles :
- Utilise TOUJOURS au moins un outil avant de repondre. Ne reponds qu'a partir des resultats des outils.
- Le document est en anglais : formule les requetes des outils en ANGLAIS.
- search_text : concepts et explications. search_tables : chiffres, scores, comparaisons.
  search_images : figures et schemas. Tu peux appeler plusieurs outils.
- N'utilise search_images que si la question porte sur une figure, un schema ou une architecture.
- Reponds dans la langue de l'utilisateur, de facon concise, et cite les pages sources, ex. (p. 3).
- Si l'information est absente des resultats, dis-le."""


@lru_cache(maxsize=1)
def get_agent():
    llm = ChatOllama(model=LLM, temperature=0, reasoning=False)
    return create_agent(llm, [search_text, search_tables, search_images], system_prompt=SYSTEM)


def ask(history: list[dict]):
    """history = [{'role': 'user'|'assistant', 'content': ...}, ...]  (gestion du contexte).
    Retourne (reponse, outils_appeles, resultats_des_outils, images) pour le dernier tour."""
    msgs = get_agent().invoke({"messages": history})["messages"]
    last = max(i for i, m in enumerate(msgs) if m.type == "human")
    calls, outputs = [], []
    for m in msgs[last + 1:]:
        if m.type == "ai":
            calls += [f"{c['name']}({c['args'].get('query', '')!r})" for c in (m.tool_calls or [])]
        elif m.type == "tool":
            outputs.append(m.content)
    images = list(dict.fromkeys(re.findall(r"IMAGE: (\S+)", "\n".join(outputs))))
    return msgs[-1].content, calls, outputs, images


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "How many attention heads does the base Transformer use?"
    answer, calls, outputs, images = ask([{"role": "user", "content": q}])
    print(f"\nOutils appeles : {calls}\n\nReponse : {answer}\n\nImages : {images}")