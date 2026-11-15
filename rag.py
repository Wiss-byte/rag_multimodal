
"""
Point d'entrée CLI du RAG multimodal.
"""

import sys

from graph_agent import ask


def main():

    if len(sys.argv) < 2:
        print(
            'Usage : uv run python rag.py '
            '"What is the Transformer architecture?"'
        )
        sys.exit(1)

    question = " ".join(sys.argv[1:])

    history = [
        {
            "role": "user",
            "content": question,
        }
    ]

    answer, calls, outputs, images = ask(history)

    print("\n" + "=" * 60)
    print("REPONSE DE L'AGENT")
    print("=" * 60)

    print(answer)

    print("\nOUTILS APPELES :")

    for call in calls:
        print(call)

    print("\nOBSERVATIONS :")

    for output in outputs:
        print(output)

    print("\nIMAGES TROUVEES :")

    for image in images:
        print(image)


if __name__ == "__main__":
    main()
