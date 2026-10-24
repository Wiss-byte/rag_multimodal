# RAG multimodal agentique

Question-réponse sur un PDF (texte, tableaux, figures) : LlamaIndex, PGVector,
CLIP, LangChain et Qwen (Ollama). Tout est open source et local.

## Prérequis
uv, Docker, Ollama.

## Installation
docker compose up -d
ollama pull qwen3:1.7b
ollama pull moondream
uv sync

## Utilisation
uv run python ingest.py data/attention.pdf          # texte + tableaux
uv run python ingest_images.py data/attention.pdf   # figures (CLIP + descriptions)
uv run python agent_rag.py "Quel score BLEU sur EN-DE ?"
uv run streamlit run app.py

## Architecture
PDF -> LlamaIndex/PyMuPDF -> embeddings (MiniLM, CLIP) -> PGVector
Question -> Streamlit -> agent LangChain (3 outils) -> PGVector -> Qwen -> réponse + sources