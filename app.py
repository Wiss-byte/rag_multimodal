
"""
Interface Streamlit pour le RAG multimodal agentique.
"""

import os
import subprocess
import sys

from pathlib import Path

import streamlit as st

from graph_agent import ask
from agent_rag import search_similar_images


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multimodal Agentic RAG",
    page_icon="📚",
    layout="wide"
)

DATA = Path("data")
IMAGE_DIR = DATA / "images"
QUERY_DIR = DATA / "query_images"

DATA.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)
QUERY_DIR.mkdir(exist_ok=True)


# ============================================================
# SESSION
# ============================================================

if "msgs" not in st.session_state:
    st.session_state.msgs = []

if "doc" not in st.session_state:
    st.session_state.doc = None


# ============================================================
# INTERFACE
# ============================================================

st.title("📚 Multimodal Agentic RAG")

st.caption(
    "Interrogez un document PDF avec un agent IA "
    "capable de rechercher du texte et des images."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("1. Document PDF")

    pdf_file = st.file_uploader(
        "Importer un PDF",
        type=["pdf"]
    )

    if pdf_file and st.button("Indexer le document"):

        pdf_path = DATA / pdf_file.name

        pdf_path.write_bytes(pdf_file.getvalue())

        env = {
            **os.environ,
            "PYTHONIOENCODING": "utf-8"
        }

        scripts = [
            "ingest.py",
            "ingest_images.py"
        ]

        success = True

        for script in scripts:

            with st.spinner(f"Exécution de {script}..."):

                result = subprocess.run(
                    [
                        sys.executable,
                        script,
                        str(pdf_path)
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=env
                )

            if result.returncode != 0:

                st.error(f"Erreur dans {script}")
                st.code(result.stderr[-3000:])

                success = False
                break

        if success:

            st.session_state.doc = pdf_file.name
            st.session_state.msgs = []

            st.success("Document indexé avec succès !")

    if st.session_state.doc:
        st.info(f"Document actif : {st.session_state.doc}")

    st.divider()

    # ========================================================
    # RECHERCHE IMAGE VERS IMAGE
    # ========================================================

    st.header("2. Recherche par image")

    uploaded_image = st.file_uploader(
        "Importer une image comme requête",
        type=["png", "jpg", "jpeg"],
        key="image_query"
    )

    k = st.slider(
        "Nombre de résultats",
        min_value=1,
        max_value=10,
        value=3
    )

    if uploaded_image:

        st.image(
            uploaded_image,
            caption="Image de recherche",
            width=250
        )

        if st.button("Rechercher les images similaires"):

            query_path = QUERY_DIR / uploaded_image.name

            query_path.write_bytes(
                uploaded_image.getvalue()
            )

            try:

                with st.spinner("Recherche CLIP en cours..."):

                    results = search_similar_images(
                        str(query_path),
                        k=k
                    )

                if not results:

                    st.warning("Aucun résultat trouvé.")

                else:

                    st.success(
                        f"{len(results)} résultats trouvés."
                    )

                    for result in results:

                        image_path = result["image_path"]

                        st.markdown(
                            f"**Page : {result['page']}**"
                        )

                        st.caption(
                            f"Similarité : {result['score']}"
                        )

                        if Path(image_path).exists():

                            st.image(
                                image_path,
                                use_container_width=True
                            )

                        st.divider()

            except Exception as error:

                st.error(f"Erreur : {error}")


# ============================================================
# HISTORIQUE DU CHAT
# ============================================================

st.header("3. Assistant documentaire")

for message in st.session_state.msgs:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        for image_path in message.get("images", []):

            if Path(image_path).exists():

                st.image(
                    image_path,
                    caption=image_path
                )


# ============================================================
# QUESTION UTILISATEUR
# ============================================================

question = st.chat_input(
    "Posez une question sur Attention Is All You Need..."
)

if question:

    st.session_state.msgs.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("L'agent réfléchit et recherche..."):

            try:

                history = [
                    {
                        "role": message["role"],
                        "content": message["content"]
                    }
                    for message in st.session_state.msgs
                ]

                answer, calls, outputs, images = ask(history)

                st.markdown(answer)

                for image_path in images:

                    if Path(image_path).exists():

                        st.image(
                            image_path,
                            caption=image_path
                        )

                with st.expander(
                    "Voir le raisonnement opérationnel et les sources"
                ):

                    st.write("Outils appelés :")

                    st.write(calls)

                    st.write("Résultats récupérés :")

                    for output in outputs:
                        st.markdown(str(output))

                st.session_state.msgs.append({
                    "role": "assistant",
                    "content": answer,
                    "images": images
                })

            except Exception as error:

                st.error(f"Une erreur est survenue : {error}")
