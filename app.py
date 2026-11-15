"""
Interface Streamlit du RAG multimodal agentique.

- Onglet « Agent » : conversation avec l'agent ReAct (LangGraph),
  image jointe possible, trace Reasoning / Action / Observation / Answer.
- Onglet « Recherche directe » : top-k par similarité cosinus, sans agent
  (texte -> texte / tableau / figure, image -> image, image -> texte).
- Barre latérale : indexation d'un PDF et choix du modèle Ollama.

Lancement : uv run streamlit run app.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path

import streamlit as st

import graph_agent
from agent_rag import retrieve, TYPE_LABELS, MIN_K, MAX_K


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG multimodal agentique",
    page_icon="📚",
    layout="wide",
)

DATA = Path("data")
IMAGE_DIR = DATA / "images"
QUERY_DIR = DATA / "query_images"

for folder in (DATA, IMAGE_DIR, QUERY_DIR):
    folder.mkdir(exist_ok=True)

EXAMPLES = [
    "Quel score BLEU le Transformer (big) obtient-il en anglais-allemand ?",
    "Explique la Figure 2 : que représentent ses deux parties ?",
    "Comment est calculé le positional encoding ?",
    "D'après la Table 3, combien de têtes d'attention utilise le modèle big ?",
]

MODALITY_LABELS = {
    "all": "Tout (texte + tableaux + meilleure figure)",
    "text": "Texte",
    "table": "Tableaux",
    "image": "Figures",
}


# ============================================================
# SESSION
# ============================================================

defaults = {
    "messages": [],          # historique de la conversation
    "pending": None,         # question d'exemple à envoyer
    "search_results": None,  # derniers résultats de la recherche directe
    "indexed_doc": None,     # dernier PDF indexé depuis l'interface
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def available_models() -> list[str]:
    """Modèles installés dans Ollama (liste vide si Ollama est injoignable)."""

    try:
        import ollama
        return sorted(model.model for model in ollama.list().models)
    except Exception:
        return []


def save_upload(uploaded_file) -> str:
    """Enregistre une image envoyée par l'utilisateur et renvoie son chemin."""

    path = QUERY_DIR / f"{int(time.time())}_{uploaded_file.name}"
    path.write_bytes(uploaded_file.getvalue())

    return path.as_posix()


def show_images(paths: list[str]) -> None:
    """Affiche des images côte à côte (3 par ligne)."""

    existing = [path for path in paths if Path(path).exists()]

    if not existing:
        return

    columns = st.columns(min(3, len(existing)))

    for index, path in enumerate(existing):
        with columns[index % len(columns)]:
            st.image(path, caption=Path(path).name, width="stretch")


def show_results(results: list[dict]) -> None:
    """Affiche les k meilleurs résultats d'une recherche directe."""

    if not results:
        st.info("Aucun résultat.")
        return

    for rank, result in enumerate(results, start=1):

        score = (
            f"{result['score']:.3f}"
            if result["score"] is not None
            else "?"
        )

        label = TYPE_LABELS.get(result["type"], result["type"])

        origin = " · référence citée" if result.get("explicit") else ""

        st.markdown(
            f"**{rank}. {label} — page {result['page']} — "
            f"score {score}{origin}**"
        )

        image_path = result.get("image_path")

        if result["type"] == "image" and image_path and Path(image_path).exists():
            st.image(image_path, caption=result["content"], width=420)
        else:
            st.text(result["content"])

        st.divider()


def show_trace(trace: list[dict]) -> None:
    """Affiche le cycle ReAct étape par étape."""

    if not trace:
        st.caption("Aucune étape enregistrée.")
        return

    for index, step in enumerate(trace, start=1):

        kind = step["step"]

        if kind == "Reasoning":
            st.markdown(f"**{index}. 🧠 Reasoning**")
            st.caption(step["content"][:1500])

        elif kind == "Action":
            st.markdown(f"**{index}. 🛠️ Action** — `{step['tool']}`")
            st.json(step["arguments"])

        elif kind == "Observation":
            st.markdown(f"**{index}. 👁️ Observation**")
            st.code(step["content"][:2500], language="text")

        elif kind == "Answer":
            st.markdown(f"**{index}. ✅ Answer**")
            st.markdown(step["content"])

        else:
            st.warning(f"{index}. {kind} — {step['content']}")


def show_assistant_message(message: dict) -> None:
    """Réponse de l'agent + figures + indicateurs + trace."""

    st.markdown(message["content"])

    show_images(message.get("images", []))

    meta = message.get("meta", {})

    if meta:
        st.caption(
            f"Modèle : {meta['model']} · "
            f"{meta['n_calls']} appel(s) d'outil · "
            f"{meta['duration']:.1f} s"
            + (" · ⚠️ génération coupée" if meta.get("truncated") else "")
        )

    with st.expander("Cycle ReAct (Reasoning → Action → Observation → Answer)"):
        show_trace(message.get("trace", []))


# ============================================================
# BARRE LATERALE
# ============================================================

with st.sidebar:

    # ---------- Document ----------
    st.header("1. Document PDF")

    pdf_file = st.file_uploader("Importer un PDF", type=["pdf"])

    if pdf_file and st.button("Indexer le document", width="stretch"):

        pdf_path = DATA / pdf_file.name
        pdf_path.write_bytes(pdf_file.getvalue())

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

        success = True

        for script in ("ingest.py", "ingest_images.py"):

            with st.spinner(f"Exécution de {script}..."):

                result = subprocess.run(
                    [sys.executable, script, str(pdf_path)],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env,
                )

            if result.returncode != 0:
                st.error(f"Erreur dans {script}")
                st.code(result.stderr[-3000:])
                success = False
                break

        if success:
            st.session_state.indexed_doc = pdf_file.name
            st.session_state.messages = []
            st.success("Document indexé.")

    if st.session_state.indexed_doc:
        st.info(f"Dernier document indexé : {st.session_state.indexed_doc}")
    else:
        st.caption("Base actuelle : data/attention.pdf (déjà indexé).")

    st.divider()

    # ---------- Agent ----------
    st.header("2. Agent")

    models = available_models()
    current_model = graph_agent.LLM_MODEL

    if models:

        options = models if current_model in models else [current_model] + models

        chosen_model = st.selectbox(
            "Modèle Ollama (MIND)",
            options,
            index=options.index(current_model),
            help="Choisir un modèle qui gère les outils (tools) et ne réfléchit pas.",
        )

        # Changement de modèle à chaud : le LLM est recréé au prochain appel
        if chosen_model != graph_agent.LLM_MODEL:
            graph_agent.LLM_MODEL = chosen_model
            graph_agent.get_mind_llm.cache_clear()

    else:
        st.warning("Ollama injoignable : vérifiez qu'il est lancé.")
        st.caption(f"Modèle configuré : {current_model}")

    if st.button("Nouvelle conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    with st.expander("Graphe de l'agent (Mermaid)"):
        st.code(graph_agent.get_agent().get_graph().draw_mermaid(), language="text")


# ============================================================
# EN-TETE
# ============================================================

st.title("📚 RAG multimodal agentique")

st.caption(
    "Article : « Attention Is All You Need » · Embeddings CLIP ViT-B-32 · "
    "PGVector · Agent ReAct LangGraph (MIND = LLM, BODY = retriever_tool)"
)

tab_agent, tab_search = st.tabs(["💬 Agent", "🔎 Recherche directe"])


# ============================================================
# ONGLET 1 : AGENT
# ============================================================

with tab_agent:

    # Questions d'exemple (tant que la conversation est vide)
    if not st.session_state.messages:

        st.markdown("**Exemples de questions :**")

        columns = st.columns(len(EXAMPLES))

        for column, example in zip(columns, EXAMPLES):
            if column.button(example, width="stretch"):
                st.session_state.pending = example
                st.rerun()

    # Historique
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            if message["role"] == "user":
                st.markdown(message["display"])
                if message.get("image"):
                    st.image(message["image"], width=220)
            else:
                show_assistant_message(message)

    # Nouvelle question (texte, avec image jointe possible)
    submitted = st.chat_input(
        "Posez une question (vous pouvez joindre une image)...",
        accept_file=True,
        file_type=["png", "jpg", "jpeg"],
    )

    question = None
    image_path = None

    if submitted:
        question = (submitted.text or "").strip()
        if submitted.files:
            image_path = save_upload(submitted.files[0])
            if not question:
                question = "À quel élément de l'article correspond cette image ?"

    elif st.session_state.pending:
        question = st.session_state.pending
        st.session_state.pending = None

    if question:

        # L'agent reçoit le chemin de l'image dans la question : il le
        # transmet à retriever_tool (argument image_path)
        content = (
            f"{question}\n\nImage fournie : {image_path}"
            if image_path
            else question
        )

        st.session_state.messages.append({
            "role": "user",
            "content": content,
            "display": question,
            "image": image_path,
        })

        with st.chat_message("user"):
            st.markdown(question)
            if image_path:
                st.image(image_path, width=220)

        with st.chat_message("assistant"):

            with st.spinner("L'agent réfléchit et cherche dans le document..."):

                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ]

                start = time.perf_counter()

                try:
                    result = graph_agent.run_agent(history)
                    error = None
                except Exception as exception:
                    result = None
                    error = exception

                duration = time.perf_counter() - start

            if error is not None:

                st.error(f"Erreur de l'agent : {error}")
                st.session_state.messages.pop()

            else:

                message = {
                    "role": "assistant",
                    "content": result["answer"],
                    "images": result["images"],
                    "trace": result["trace"],
                    "meta": {
                        "model": result["model"],
                        "n_calls": len(result["calls"]),
                        "duration": duration,
                        "truncated": result.get("truncated", False),
                    },
                }

                show_assistant_message(message)

                st.session_state.messages.append(message)


# ============================================================
# ONGLET 2 : RECHERCHE DIRECTE (sans agent)
# ============================================================

with tab_search:

    st.markdown(
        "Recherche par **similarité cosinus** dans l'espace CLIP, "
        "sans passer par l'agent. La requête peut être un texte ou une image."
    )

    query_type = st.radio(
        "Type de requête",
        ["Texte", "Image"],
        horizontal=True,
    )

    k = st.slider("Nombre de résultats (k)", MIN_K, MAX_K, 5)

    if query_type == "Texte":

        query = st.text_input(
            "Requête",
            placeholder="ex. BLEU score of the big Transformer",
            help="CLIP a été entraîné en anglais : une requête en anglais donne de meilleurs résultats.",
        )

        modality = st.selectbox(
            "Rechercher dans",
            list(MODALITY_LABELS),
            format_func=MODALITY_LABELS.get,
        )

        if st.button("Rechercher", key="text_search") and query.strip():
            try:
                with st.spinner("Recherche CLIP..."):
                    st.session_state.search_results = retrieve(
                        query=query,
                        modality=modality,
                        k=k,
                    )
            except Exception as error:
                st.error(f"Erreur : {error}")

    else:

        source = st.radio(
            "Source de l'image",
            ["Figure du document", "Importer une image"],
            horizontal=True,
        )

        query_image = None

        if source == "Figure du document":

            figures = sorted(path.as_posix() for path in IMAGE_DIR.glob("*.png"))

            if figures:
                query_image = st.selectbox("Figure", figures)
            else:
                st.warning("Aucune figure indexée dans data/images.")

        else:

            uploaded = st.file_uploader(
                "Image requête",
                type=["png", "jpg", "jpeg"],
                key="search_image",
            )

            if uploaded:
                query_image = save_upload(uploaded)

        target = st.selectbox(
            "Rechercher",
            ["image", "text"],
            format_func={
                "image": "Figures similaires (image → image)",
                "text": "Passages liés (image → texte)",
            }.get,
        )

        if query_image:
            st.image(query_image, caption="Image requête", width=250)

        if st.button("Rechercher", key="image_search") and query_image:
            try:
                with st.spinner("Recherche CLIP..."):
                    st.session_state.search_results = retrieve(
                        image_path=query_image,
                        modality=target,
                        k=k,
                    )
            except Exception as error:
                st.error(f"Erreur : {error}")

    if st.session_state.search_results is not None:
        st.subheader("Résultats")
        show_results(st.session_state.search_results)