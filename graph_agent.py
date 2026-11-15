
"""
AGENT REACT MULTIMODAL.

MIND : Qwen via Ollama
BODY : outils de recherche multimodale
Orchestration : LangChain Agent Runtime

LangChain create_agent utilise un runtime basé sur LangGraph.
"""

from functools import lru_cache

from langchain.agents import create_agent
from langchain_ollama import ChatOllama

from agent_rag import TOOLS, extract_images


# ============================================================
# CONFIGURATION
# ============================================================

LLM_MODEL = "qwen3:1.7b"

SYSTEM_PROMPT = """
Tu es un assistant intelligent spécialisé dans
l'analyse du document scientifique :

Attention Is All You Need.

Tu dois répondre uniquement à partir des résultats
récupérés dans la base documentaire.

REGLES DE FONCTIONNEMENT :

1. Utilise au moins un outil avant de répondre.
2. Analyse la question avant de sélectionner les outils.
3. search_text :
   Pour les concepts, définitions et explications.
4. search_tables :
   Pour les résultats numériques et les tableaux.
5. search_images :
   Pour les figures, architectures et diagrammes.
6. Tu peux appeler plusieurs outils si nécessaire.
7. Analyse les observations retournées avant de répondre.
8. Si les résultats sont insuffisants, indique-le clairement.
9. Réponds dans la langue de l'utilisateur.
10. Cite les pages du document dans ta réponse.
11. N'invente aucune information.

Pour une question nécessitant plusieurs modalités,
combine les résultats textuels et visuels.
"""


# ============================================================
# INITIALISATION DU LLM
# ============================================================

@lru_cache(maxsize=1)
def get_llm():

    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
    )


# ============================================================
# INITIALISATION DE L'AGENT
# ============================================================

@lru_cache(maxsize=1)
def get_agent():

    llm = get_llm()

    agent = create_agent(
        model=llm,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


# ============================================================
# EXECUTION DE L'AGENT
# ============================================================

def ask(history: list[dict]):
    """
    Exécute l'agent sur l'historique conversationnel.

    Retourne :
    - answer : réponse finale
    - calls : outils appelés
    - outputs : observations des outils
    - images : chemins des images retrouvées
    """

    if not history:
        raise ValueError("L'historique ne peut pas être vide.")

    result = get_agent().invoke(
        {
            "messages": history
        }
    )

    messages = result["messages"]

    calls = []
    outputs = []

    # Récupération des appels et observations
    for message in messages:

        # Appels des outils par le LLM
        tool_calls = getattr(
            message,
            "tool_calls",
            [],
        )

        for call in tool_calls:

            calls.append({
                "tool": call.get("name"),
                "arguments": call.get("args", {}),
            })

        # Résultats retournés par les outils
        if getattr(message, "type", "") == "tool":

            outputs.append({
                "tool": getattr(message, "name", "unknown"),
                "content": message.content,
            })

    # Dernière réponse de l'assistant
    answer = ""

    for message in reversed(messages):

        if (
            getattr(message, "type", "") == "ai"
            and not getattr(message, "tool_calls", [])
        ):

            content = message.content

            if isinstance(content, str):
                answer = content

            elif isinstance(content, list):
                answer = "\n".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict)
                )

            if answer:
                break

    if not answer:
        answer = "L'agent n'a pas généré de réponse finale."

    images = extract_images(
        [item["content"] for item in outputs]
    )

    return answer, calls, outputs, images
