"""
EVALUATION DE L'AGENT RAG MULTIMODAL (étape 7).

Soumet une série de requêtes (texte, tableau, figure, hybride, image,
hors sujet) à l'agent, enregistre le cycle ReAct complet et calcule
des indicateurs simples :

- tool_called   : l'agent a-t-il appelé retriever_tool ?
- modality_ok   : a-t-il choisi une modalité attendue ?
- retrieval_ok  : la bonne page figure-t-elle dans les observations ?
- answer_ok     : la réponse contient-elle les éléments attendus ?

Sorties : results/eval_<modele>_<date>.json et .md

Usage :
    uv run python evaluate_agent.py
    uv run python evaluate_agent.py --model qwen3:1.7b
    uv run python evaluate_agent.py --only q2 q5
"""

import os
import re
import json
import time
import argparse
from pathlib import Path
from datetime import datetime


RESULTS_DIR = Path("results")
IMAGE_DIR = Path("data/images")

OBSERVATION_PREVIEW = 500   # caractères d'observation gardés dans le rapport

# Marqueurs d'une réponse « je ne sais pas » (question hors document)
REFUSAL_MARKERS = [
    "ne contient pas", "ne mentionne pas", "pas mentionné",
    "pas d'information", "aucune information", "n'est pas indiqué",
    "ne permet pas", "pas disponible", "not mention", "does not",
    "no information", "not provided", "not available", "cannot",
]


# ============================================================
# JEU DE REQUETES
# ============================================================

def find_query_image() -> str | None:
    """Image utilisée pour la requête par image (Figure 2, partie gauche)."""

    matches = sorted(IMAGE_DIR.glob("*_p4_img2.png"))

    return matches[0].as_posix() if matches else None


def build_test_cases() -> list[dict]:
    """
    Chaque cas précise :
    - expected_modalities : modalités jugées pertinentes
    - expected_pages      : pages où se trouve la réponse
    - keywords            : groupes de mots attendus dans la réponse
                            (un groupe est validé si l'une des variantes apparaît)
    """

    cases = [
        {
            "id": "q1",
            "category": "texte",
            "question": (
                "What is the role of positional encoding in the "
                "Transformer, and which functions are used to compute it?"
            ),
            "expected_modalities": ["text", "all"],
            "expected_pages": [6],
            "keywords": [["sine", "sinus", "sinusoid"], ["cosine", "cosinus"]],
        },
        {
            "id": "q2",
            "category": "tableau",
            "question": (
                "Quel score BLEU le Transformer (big) obtient-il "
                "sur la traduction anglais-allemand ?"
            ),
            "expected_modalities": ["table", "all"],
            "expected_pages": [8],
            "keywords": [["28.4", "28,4"]],
        },
        {
            "id": "q3",
            "category": "tableau",
            "question": (
                "According to Table 3, how many attention heads and "
                "which d_model does the big model use?"
            ),
            "expected_modalities": ["table", "all"],
            "expected_pages": [9],
            "keywords": [["16"], ["1024"]],
        },
        {
            "id": "q4",
            "category": "tableau (multi-étapes)",
            "question": (
                "Compare the training cost in FLOPs on English-to-German "
                "of the big Transformer and of the ConvS2S Ensemble."
            ),
            "expected_modalities": ["table", "all"],
            "expected_pages": [8],
            "keywords": [["2.3"], ["7.7"]],
        },
        {
            "id": "q5",
            "category": "figure",
            "question": "Que représente la Figure 1 de l'article ?",
            "expected_modalities": ["image", "all"],
            "expected_pages": [3],
            "keywords": [["encoder", "encodeur"], ["decoder", "décodeur"]],
        },
        {
            "id": "q6",
            "category": "hybride texte + figure",
            "question": (
                "Explique la Figure 2 : que représentent ses deux parties, "
                "et comment fonctionne la multi-head attention ?"
            ),
            "expected_modalities": ["all", "image"],
            "expected_pages": [4],
            "keywords": [
                ["scaled dot-product", "produit scalaire", "dot-product"],
                ["multi-head", "multi-tête", "multi-têtes", "plusieurs têtes"],
            ],
        },
        {
            "id": "q8",
            "category": "hors document",
            "question": "Quel score BLEU obtient GPT-4 sur WMT 2014 anglais-allemand ?",
            "expected_modalities": ["table", "text", "all"],
            "expected_pages": [],
            "keywords": [],
            "expect_refusal": True,
        },
    ]

    image = find_query_image()

    if image:
        cases.insert(6, {
            "id": "q7",
            "category": "requête par image",
            "question": (
                f"Voici une image : {image}. "
                "À quel mécanisme de l'article correspond-elle ?"
            ),
            "expected_modalities": ["image", "text", "all"],
            "expected_pages": [4],
            "keywords": [["scaled dot-product", "produit scalaire", "dot-product"]],
            "expect_image_path": True,
        })

    return cases


# ============================================================
# INDICATEURS
# ============================================================

def pages_in_observations(outputs: list[dict]) -> list[int]:
    """Pages citées dans les observations (« page 8 » ou « Page: 8 »)."""

    pages = set()

    for output in outputs:
        for number in re.findall(
            r"[Pp]age:?\s*(\d+)",
            output["content"],
        ):
            pages.add(int(number))

    return sorted(pages)


def contains(answer: str, variant: str) -> bool:
    """
    Recherche d'un mot attendu dans la réponse.
    Pour un nombre, on évite les faux positifs (« 16 » dans « 2016 »).
    """

    if re.fullmatch(r"[\d.,]+", variant):
        pattern = rf"(?<![\d.,]){re.escape(variant)}(?!\d)"
        return re.search(pattern, answer) is not None

    return variant.lower() in answer.lower()


def keyword_group_found(answer: str, group: list[str]) -> bool:
    """Un groupe est validé si l'une de ses variantes apparaît."""
    return any(contains(answer, variant) for variant in group)


def evaluate_case(case: dict, result: dict, duration: float) -> dict:

    answer = result["answer"]
    calls = result["calls"]
    outputs = result["outputs"]

    modalities = [
        call["arguments"].get("modality", "all")
        for call in calls
    ]

    used_image_path = any(
        call["arguments"].get("image_path")
        for call in calls
    )

    found_pages = pages_in_observations(outputs)

    checks = {
        "tool_called": bool(calls),
        "n_calls": len(calls),
        "modalities": modalities,
        "modality_ok": any(
            modality in case["expected_modalities"]
            for modality in modalities
        ),
        "observed_pages": found_pages,
        "retrieval_ok": (
            any(page in found_pages for page in case["expected_pages"])
            if case["expected_pages"]
            else None
        ),
        "duration_s": round(duration, 1),
        "images": result["images"],
    }

    if case.get("expect_refusal"):
        checks["answer_ok"] = any(
            marker in answer.lower()
            for marker in REFUSAL_MARKERS
        )
    else:
        missing = [
            group[0]
            for group in case["keywords"]
            if not keyword_group_found(answer, group)
        ]
        checks["answer_ok"] = not missing
        checks["missing_keywords"] = missing

    if case.get("expect_image_path"):
        checks["image_path_used"] = used_image_path

    return checks


# ============================================================
# RAPPORTS
# ============================================================

def mark(value) -> str:
    if value is None:
        return "–"
    return "✅" if value else "❌"


def rate(records: list[dict], key: str) -> str:
    values = [r["checks"][key] for r in records if r["checks"].get(key) is not None]
    if not values:
        return "–"
    return f"{sum(values)}/{len(values)}"


def build_markdown(model: str, records: list[dict], mermaid: str) -> str:

    lines = [
        f"# Évaluation de l'agent RAG multimodal — `{model}`",
        "",
        f"Date : {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Graphe de l'agent (LangGraph)",
        "",
        "```mermaid",
        mermaid.strip(),
        "```",
        "",
        "## Synthèse",
        "",
        "| Id | Catégorie | Outil appelé | Appels | Modalités | Modalité OK | "
        "Bonne page retrouvée | Réponse OK | Durée (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for record in records:

        checks = record["checks"]

        lines.append(
            f"| {record['id']} | {record['category']} | "
            f"{mark(checks['tool_called'])} | {checks['n_calls']} | "
            f"{', '.join(checks['modalities']) or '–'} | "
            f"{mark(checks['modality_ok'])} | "
            f"{mark(checks['retrieval_ok'])} | "
            f"{mark(checks['answer_ok'])} | {checks['duration_s']} |"
        )

    durations = [r["checks"]["duration_s"] for r in records]

    lines += [
        "",
        f"- Appel d'outil : **{rate(records, 'tool_called')}**",
        f"- Modalité pertinente : **{rate(records, 'modality_ok')}**",
        f"- Bonne page retrouvée : **{rate(records, 'retrieval_ok')}**",
        f"- Réponse correcte : **{rate(records, 'answer_ok')}**",
        f"- Durée moyenne : **{sum(durations) / max(len(durations), 1):.1f} s**",
        "",
        "## Détail des cycles ReAct",
        "",
    ]

    for record in records:

        checks = record["checks"]

        lines += [
            f"### {record['id']} — {record['category']}",
            "",
            f"**Question :** {record['question']}",
            "",
        ]

        for index, step in enumerate(record["trace"], start=1):

            if step["step"] == "Action":
                arguments = json.dumps(step["arguments"], ensure_ascii=False)
                lines.append(f"{index}. **Action** — `{step['tool']}({arguments})`")

            elif step["step"] == "Observation":
                preview = step["content"][:OBSERVATION_PREVIEW]
                if len(step["content"]) > OBSERVATION_PREVIEW:
                    preview += " [...]"
                lines += [
                    f"{index}. **Observation**",
                    "",
                    "```text",
                    preview,
                    "```",
                ]

            else:
                lines += [f"{index}. **{step['step']}**", "", step["content"]]

            lines.append("")

        lines += [
            f"**Indicateurs :** outil {mark(checks['tool_called'])} · "
            f"modalité {mark(checks['modality_ok'])} · "
            f"page {mark(checks['retrieval_ok'])} · "
            f"réponse {mark(checks['answer_ok'])} · "
            f"{checks['duration_s']} s",
        ]

        if checks.get("missing_keywords"):
            lines.append(
                f"  — éléments absents de la réponse : "
                f"{', '.join(checks['missing_keywords'])}"
            )

        if "image_path_used" in checks:
            lines.append(
                f"  — image transmise à l'outil : {mark(checks['image_path_used'])}"
            )

        if checks["images"]:
            lines.append(f"  — images affichées : {', '.join(checks['images'])}")

        lines += ["", "---", ""]

    return "\n".join(lines)


# ============================================================
# EXECUTION
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Évaluation de l'agent RAG multimodal."
    )

    parser.add_argument("--model", default=None, help="Modèle Ollama à tester")
    parser.add_argument("--only", nargs="*", default=None, help="Ids à exécuter")

    args = parser.parse_args()

    # Le modèle doit être choisi AVANT d'importer l'agent
    if args.model:
        os.environ["RAG_LLM"] = args.model

    from graph_agent import run_agent, get_agent, LLM_MODEL

    cases = build_test_cases()

    if args.only:
        cases = [case for case in cases if case["id"] in args.only]

    RESULTS_DIR.mkdir(exist_ok=True)

    records = []

    print(f"Modèle : {LLM_MODEL} — {len(cases)} requêtes\n")

    for number, case in enumerate(cases, start=1):

        print(f"[{number}/{len(cases)}] {case['id']} ({case['category']}) ...", flush=True)

        start = time.perf_counter()

        result = run_agent([{"role": "user", "content": case["question"]}])

        duration = time.perf_counter() - start

        checks = evaluate_case(case, result, duration)

        records.append({
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "answer": result["answer"],
            "trace": result["trace"],
            "checks": checks,
        })

        print(
            f"    outil {mark(checks['tool_called'])}  "
            f"modalité {mark(checks['modality_ok'])} {checks['modalities']}  "
            f"page {mark(checks['retrieval_ok'])}  "
            f"réponse {mark(checks['answer_ok'])}  "
            f"{checks['duration_s']} s"
        )

    stamp = f"{datetime.now():%Y%m%d-%H%M}"
    model_slug = re.sub(r"[^a-zA-Z0-9-]+", "-", LLM_MODEL)
    name = f"eval_{model_slug}_{stamp}"

    json_path = RESULTS_DIR / f"{name}.json"
    markdown_path = RESULTS_DIR / f"{name}.md"

    json_path.write_text(
        json.dumps(
            {"model": LLM_MODEL, "date": stamp, "records": records},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    mermaid = get_agent().get_graph().draw_mermaid()

    markdown_path.write_text(
        build_markdown(LLM_MODEL, records, mermaid),
        encoding="utf-8",
    )

    print("\nSynthèse :")
    print(f"  appel d'outil        {rate(records, 'tool_called')}")
    print(f"  modalité pertinente  {rate(records, 'modality_ok')}")
    print(f"  bonne page retrouvée {rate(records, 'retrieval_ok')}")
    print(f"  réponse correcte     {rate(records, 'answer_ok')}")
    print(f"\nRapport : {markdown_path}")
    print(f"Données : {json_path}")


if __name__ == "__main__":
    main()