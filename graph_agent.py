
"""
Agent RAG multimodal avec LangGraph.
Pattern ReAct : Reasoning -> Action -> Observation.
"""

from functools import lru_cache

from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition

from agent_rag import TOOLS


LLM = "qwen3:1.7b"

SYSTEM = """
Tu es un assistant intelligent spécialisé dans
l'analyse du document scientifique Attention Is All You Need.

Tu disposes de trois outils :

1. search_text :
   Recherche les concepts et explications.

2. search_tables :
   Recherche les résultats numériques et tableaux.

3. search_images :
   Recherche les figures, schémas et architectures.

REGLES :

- Utilise les outils avant de répondre.
- Tu peux utiliser plusieurs outils pour une question.
- Formule les requêtes de recherche en anglais.
- Base tes réponses sur les résultats récupérés.
- Cite les numéros de pages.
- Réponds dans la langue de l'utilisateur.
- Si une information est absente, indique-le.
"""


@lru_cache(maxsize=1)
def build_graph():

    # MIND : le modèle de langage
    llm = ChatOllama(
        model=LLM,
        temperature=0,
    ).bind_tools(TOOLS)

    # Fonction de raisonnement
    def agent_node(state: MessagesState):

        messages = state["messages"]

        response = llm.invoke(
            [{"role": "system", "content": SYSTEM}]
            + messages
        )

        return {
            "messages": [response]
        }

    # Construction du graphe
    workflow = StateGraph(MessagesState)

    workflow.add_node(
        "agent",
        agent_node
    )

    # BODY : les outils
    workflow.add_node(
        "tools",
        ToolNode(TOOLS)
    )

    workflow.add_edge(
        START,
        "agent"
    )

    # Décision : appeler un outil ou terminer
    workflow.add_conditional_edges(
        "agent",
        tools_condition
    )

    # Retour à l'agent après observation
    workflow.add_edge(
        "tools",
        "agent"
    )

    return workflow.compile()


def ask(history):

    graph = build_graph()

    result = graph.invoke({
        "messages": history
    })

    messages = result["messages"]

    last_user = max(
        i for i, message in enumerate(messages)
        if message.type == "human"
    )

    calls = []
    outputs = []

    for message in messages[last_user + 1:]:

        if message.type == "ai":

            for call in message.tool_calls or []:

                calls.append(
                    f"{call['name']}({call['args']})"
                )

        elif message.type == "tool":

            outputs.append(message.content)

    answer = messages[-1].content

    from agent_rag import extract_images

    images = extract_images(outputs)

    return answer, calls, outputs, images
