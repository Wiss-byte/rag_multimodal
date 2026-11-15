"""
AGENT REACT MULTIMODAL AVEC LANGGRAPH (graphe explicite).

MIND : LLM Qwen3 via Ollama (raisonne et décide)
BODY : ToolNode qui exécute retriever_tool (recherche multimodale)

Cycle ReAct :
    START -> mind -> (appel d'outil ?) -> body -> mind -> ... -> END
              Reasoning      Action        Observation      Réponse
"""

import os
import re
import json
import argparse
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.errors import GraphRecursionError

from agent_rag import TOOLS


# ============================================================
# CONFIGURATION
# ============================================================

# Modèle Ollama : modifiable sans toucher au code
#   PowerShell : $env:RAG_LLM = "llama3.2"
LLM_MODEL = os.getenv("RAG_LLM", "qwen3:1.7b")

# True : Qwen3 « réfléchit » avant d'agir (plus lent, raisonnement visible)
REASONING = False

# Réglages Ollama pour une machine sans GPU (tout tourne sur le processeur)
# Attention : avec un modèle « thinking » (ex. qwen3:4b), la réflexion
# consomme NUM_PREDICT avant tout appel d'outil -> préférer une variante instruct
NUM_CTX = 8192        # contexte assez grand pour garder prompt + observations
NUM_PREDICT = 512     # longueur max d'une réponse : borne le temps de génération
KEEP_ALIVE = "30m"    # garde le modèle en mémoire entre deux questions

# Garde-fou : nombre maximal d'étapes du graphe (mind + body)
RECURSION_LIMIT = 10

SYSTEM_PROMPT = """Tu es un assistant de recherche sur l'article scientifique
« Attention Is All You Need ».

Méthode (ReAct) :
1. Réfléchis à ce que demande la question.
2. Appelle retriever_tool avec une query en anglais et la modality adaptée :
   - chiffres, scores, résultats, hyperparamètres -> modality="table"
   - définitions, explications, méthodes          -> modality="text"
   - figures, schémas, architecture visuelle      -> modality="image"
   - question qui mélange explication et figure   -> modality="all"
   Si la question cite « Figure N » ou « Table N », garde-le dans query.
   Si l'utilisateur donne le chemin d'une image, passe-le dans image_path.
3. Lis les résultats. S'ils ne suffisent pas, refais une recherche avec
   une autre requête ou une autre modality (3 recherches au maximum).
4. Réponds.

Exemples d'appels :
- « Combien de couches a l'encodeur ? »
  -> retriever_tool(query="number of encoder layers N", modality="text")
- « Quel est le coût d'entraînement de ConvS2S ? »
  -> retriever_tool(query="training cost FLOPs ConvS2S", modality="table")
- « Que montre la Figure 1 ? »
  -> retriever_tool(query="Figure 1 Transformer model architecture", modality="all")

Règles de réponse :
- réponds uniquement à partir des résultats de retriever_tool, n'invente rien ;
- recopie les chiffres exactement comme dans les résultats,
  en lisant la bonne ligne du tableau ;
- cite les pages (ex. « page 8 ») et les tableaux ou figures utilisés ;
- ne cite jamais les scores de similarité : ils servent seulement au classement ;
- si les résultats ne permettent pas de répondre, dis-le clairement ;
- réponds dans la langue de l'utilisateur."""


# ============================================================
# ETAT DU GRAPHE
# ============================================================

class AgentState(TypedDict):
    """Mémoire de l'agent : la liste des messages échangés."""

    messages: Annotated[list[AnyMessage], add_messages]


# ============================================================
# MIND : le LLM
# ============================================================

@lru_cache(maxsize=1)
def get_mind_llm():
    """LLM capable d'appeler les outils (bind_tools)."""

    llm = ChatOllama(
        model=LLM_MODEL,
        temperature=0,
        reasoning=REASONING,
        num_ctx=NUM_CTX,
        num_predict=NUM_PREDICT,
        keep_alive=KEEP_ALIVE,
    )

    return llm.bind_tools(TOOLS)


def mind(state: AgentState) -> dict:
    """
    Noeud MIND : à partir de la conversation et des observations,
    le LLM décide d'appeler un outil (Action) ou de répondre.
    """

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    response = get_mind_llm().invoke(messages)

    return {"messages": [response]}


# ============================================================
# BODY : les outils
# ============================================================

body = ToolNode(TOOLS)


# ============================================================
# ROUTAGE
# ============================================================

def route_after_mind(state: AgentState) -> str:
    """Action demandée -> BODY ; sinon -> fin (réponse finale)."""

    last_message = state["messages"][-1]

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "body"

    return END


# ============================================================
# COMPOSITION ET COMPILATION DU GRAPHE
# ============================================================

def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("mind", mind)
    graph.add_node("body", body)

    graph.add_edge(START, "mind")

    graph.add_conditional_edges(
        "mind",
        route_after_mind,
        {
            "body": "body",
            END: END,
        },
    )

    # Après l'observation, retour au cerveau : boucle ReAct
    graph.add_edge("body", "mind")

    return graph.compile()


@lru_cache(maxsize=1)
def get_agent():
    return build_graph()


# ============================================================
# OUTILS D'ANALYSE DES MESSAGES
# ============================================================

def split_reasoning(message) -> tuple[str, str]:
    """
    Sépare la réflexion du texte visible : (texte, réflexion).

    Gère les trois formes produites par Qwen3 :
    - « <think> ... </think> réponse »
    - « ... </think> réponse » (variante thinking : <think> implicite)
    - « <think> ... » sans fermeture (réflexion coupée)
    """

    content = message.content

    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict)
        )

    content = str(content)
    reasoning = ""

    if "</think>" in content:
        reasoning, content = content.rsplit("</think>", 1)
        reasoning = reasoning.replace("<think>", "")

    elif "<think>" in content:
        content, reasoning = content.split("<think>", 1)

    return content.strip(), reasoning.strip()


def images_with_labels(observation: str) -> list[tuple[str, str]]:
    """Couples (chemin image, « Figure N ») présents dans une observation."""

    return re.findall(
        r"IMAGE:\s*(.+?)\s*\|\s*Page:[^\n]*\n(Figure\s+\d+)?",
        observation,
    )


def select_images(answer: str, observations: list[dict]) -> list[str]:
    """
    Images à afficher :
    - toutes, si l'agent a explicitement cherché des figures
      (modality="image" ou recherche par image) ;
    - sinon, seulement celles dont la figure est citée dans la réponse.
    """

    answer_lower = answer.lower()

    selected = []

    for observation in observations:

        args = observation["arguments"]

        explicit = (
            args.get("modality") == "image"
            or bool(args.get("image_path"))
        )

        for path, label in images_with_labels(observation["content"]):

            if explicit or (label and label.lower() in answer_lower):
                selected.append(path)

    return list(dict.fromkeys(selected))


# ============================================================
# EXECUTION DE L'AGENT AVEC TRACE ReAct
# ============================================================

def run_agent(history: list[dict]) -> dict:
    """
    Exécute le graphe et retourne :
    - answer : réponse finale
    - trace : étapes Reasoning / Action / Observation / Answer
    - calls : outils appelés
    - outputs : observations des outils
    - images : images à afficher
    - model : modèle utilisé
    - truncated : True si une génération a été coupée (NUM_PREDICT)
    """

    if not history:
        raise ValueError("L'historique ne peut pas être vide.")

    trace = []
    calls = []
    outputs = []
    call_arguments = {}
    answer = ""
    truncated = False

    try:

        for update in get_agent().stream(
            {"messages": history},
            config={"recursion_limit": RECURSION_LIMIT},
            stream_mode="updates",
        ):

            for node_name, node_output in update.items():

                for message in node_output.get("messages", []):

                    # ---------- MIND ----------
                    if node_name == "mind" and isinstance(message, AIMessage):

                        text, inline_reasoning = split_reasoning(message)

                        reasoning = (
                            message.additional_kwargs.get("reasoning_content", "")
                            or inline_reasoning
                        )

                        if reasoning:
                            trace.append({
                                "step": "Reasoning",
                                "content": reasoning.strip(),
                            })

                        done_reason = message.response_metadata.get(
                            "done_reason",
                        )

                        if message.tool_calls:

                            for call in message.tool_calls:

                                call_arguments[call["id"]] = call["args"]

                                calls.append({
                                    "tool": call["name"],
                                    "arguments": call["args"],
                                })

                                trace.append({
                                    "step": "Action",
                                    "tool": call["name"],
                                    "arguments": call["args"],
                                })

                        elif done_reason == "length":

                            # Le modèle a épuisé NUM_PREDICT (souvent en
                            # réfléchissant) avant de décider ou de répondre
                            truncated = True

                            trace.append({
                                "step": "Stop",
                                "content": (
                                    f"Génération coupée : limite de "
                                    f"{NUM_PREDICT} tokens atteinte avant "
                                    f"une décision ou une réponse complète."
                                ),
                            })

                            if text:
                                answer = f"{text} [réponse tronquée]"

                        else:

                            answer = text

                            trace.append({
                                "step": "Answer",
                                "content": answer,
                            })

                    # ---------- BODY ----------
                    if node_name == "body" and isinstance(message, ToolMessage):

                        content = str(message.content)

                        outputs.append({
                            "tool": message.name,
                            "arguments": call_arguments.get(
                                message.tool_call_id,
                                {},
                            ),
                            "content": content,
                        })

                        trace.append({
                            "step": "Observation",
                            "tool": message.name,
                            "content": content,
                        })

    except GraphRecursionError:

        answer = (
            "L'agent a atteint la limite d'étapes sans produire "
            "de réponse finale."
        )

        trace.append({"step": "Stop", "content": answer})

    if not answer:
        answer = "L'agent n'a pas généré de réponse finale."

    return {
        "model": LLM_MODEL,
        "answer": answer,
        "truncated": truncated,
        "trace": trace,
        "calls": calls,
        "outputs": outputs,
        "images": select_images(answer, outputs),
    }


def ask(history: list[dict]):
    """Interface utilisée par app.py et rag.py."""

    result = run_agent(history)

    return (
        result["answer"],
        result["calls"],
        result["outputs"],
        result["images"],
    )


def format_trace(trace: list[dict], max_chars: int = 600) -> str:
    """Affichage lisible du cycle ReAct."""

    lines = []

    for index, step in enumerate(trace, start=1):

        if step["step"] == "Action":

            arguments = json.dumps(step["arguments"], ensure_ascii=False)

            lines.append(
                f"{index}. ACTION      -> {step['tool']}({arguments})"
            )

        else:

            content = step["content"]

            if len(content) > max_chars:
                content = content[:max_chars] + " [...]"

            lines.append(f"{index}. {step['step'].upper():<11} -> {content}")

    return "\n\n".join(lines)


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Agent RAG multimodal (LangGraph)."
    )

    parser.add_argument(
        "--mermaid",
        action="store_true",
        help="Affiche le graphe de l'agent au format Mermaid",
    )

    args = parser.parse_args()

    if args.mermaid:
        print(get_agent().get_graph().draw_mermaid())
    else:
        parser.print_help()