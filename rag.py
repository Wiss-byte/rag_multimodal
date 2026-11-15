"""
Point d'entrée CLI de l'agent RAG multimodal.
Affiche le cycle ReAct (Action / Observation / Answer) puis la réponse.
"""

import sys

from graph_agent import run_agent, format_trace


def main():

    if len(sys.argv) < 2:
        print(
            'Usage : uv run python rag.py '
            '"What is the Transformer architecture?"'
        )
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    result = run_agent([
        {
            "role": "user",
            "content": question,
        }
    ])

    print("=" * 60)
    print("CYCLE ReAct")
    print("=" * 60)

    print(format_trace(result["trace"]))

    print("\n" + "=" * 60)
    print("REPONSE DE L'AGENT")
    print("=" * 60)

    print(result["answer"])

    if result["images"]:

        print("\nIMAGES A AFFICHER :")

        for image in result["images"]:
            print(" -", image)


if __name__ == "__main__":
    main()