"""Interface Streamlit : charger un PDF, l'indexer, poser des questions a l'agent."""
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="RAG multimodal agentique", layout="wide")
st.title("RAG multimodal agentique")

DATA = Path("data")
DATA.mkdir(exist_ok=True)

if "msgs" not in st.session_state:
    st.session_state.msgs = []
    st.session_state.doc = None

# ---------------------------------------------------------------- barre laterale : chargement du PDF
with st.sidebar:
    st.header("Document")
    up = st.file_uploader("Charger un PDF", type="pdf")
    if up and st.button("Indexer ce PDF"):
        path = DATA / up.name
        path.write_bytes(up.getvalue())
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        with st.spinner("Indexation : texte, tableaux, images (plusieurs minutes)..."):
            for script in ("ingest.py", "ingest_images.py"):
                r = subprocess.run([sys.executable, script, str(path)], capture_output=True,
                                   text=True, encoding="utf-8", errors="replace", env=env)
                if r.returncode != 0:
                    st.error(f"Erreur dans {script}")
                    st.code(r.stderr[-1500:])
                    break
            else:
                st.session_state.doc, st.session_state.msgs = up.name, []
                st.success("Indexation terminee.")
    if st.session_state.doc:
        st.info(f"Document actif : {st.session_state.doc}")
    else:
        st.caption("Aucun PDF charge dans cette session. Si la base est deja remplie, vous pouvez poser vos questions.")

# ---------------------------------------------------------------- historique
for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for img in m.get("images", []):
            st.image(img, caption=img)

# ---------------------------------------------------------------- nouvelle question
if q := st.chat_input("Posez votre question sur le document..."):
    st.session_state.msgs.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.spinner("L'agent cherche et reflechit..."):
            from agent_rag import ask
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs]
            answer, calls, outputs, images = ask(history)
        st.markdown(answer)
        for img in images:
            st.image(img, caption=img)
        with st.expander("Outils utilises et sources recuperees"):
            st.write("Outils appeles :", calls)
            for o in outputs:
                st.markdown(o)
    st.session_state.msgs.append({"role": "assistant", "content": answer, "images": images})