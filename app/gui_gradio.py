from __future__ import annotations

import logging
from typing import Tuple

import gradio as gr

from pipelines.retrieval_pipeline import get_default_retrieval_pipeline

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# On initialise le pipeline trustRAG une seule fois
pipeline = get_default_retrieval_pipeline()


def answer_query(question: str) -> Tuple[str, str, str]:
    """
    Fonction appelée par Gradio.

    Retourne :
      - réponse finale (LLM + guardrails)
      - résumé du grounding_check
      - résumé du contexte utilisé
    """
    result = pipeline.answer_question(question)

    final_answer = result.get("final_answer", "")
    grounding_check = result.get("grounding_check", None)
    context_summary = result.get("used_context_summary", "")

    # Construire un petit rapport lisible sur le grounding
    if grounding_check is None:
        grounding_report = "Aucun contrôle de grounding (question vide ?)."
    else:
        try:
            reasons_str = "\n".join(f"- {r}" for r in grounding_check.reasons)
        except Exception:
            reasons_str = str(getattr(grounding_check, "reasons", ""))

        max_auth = getattr(grounding_check, "max_authority_score", None)
        overlap = getattr(grounding_check, "lexical_overlap", None)

        grounding_report = (
            "### Grounding check\n"
            f"- is_grounded = {grounding_check.is_grounded}\n"
        )

        if max_auth is not None:
            grounding_report += f"- max_authority_score = {max_auth:.2f}\n"
        if overlap is not None:
            grounding_report += f"- lexical_overlap = {overlap:.2f}\n"

        grounding_report += "\n**Raisons :**\n" + reasons_str

    return final_answer, grounding_report, context_summary


# ----------------------------------------------------------------------
#  Gradio UI
# ----------------------------------------------------------------------
with gr.Blocks(title="trustRAG Demo") as demo:
    gr.Markdown(
        """
        # 🛡️ trustRAG – Demo

        Pose une question sur les filings SEC (Apple 10-K 2025, etc.).
        Le système utilise :
        - RAG Fusion (reformulation via LLM Ollama)
        - Retrieval hybride (vectoriel + future KG)
        - Reranking par score d'autorité
        - Grounding guardrails (refus si pas assez ancré)
        """
    )

    with gr.Row():
        question = gr.Textbox(
            label="Question utilisateur",
            placeholder="Ex: Quel est le montant de la dette à long terme d'Apple en 2025 ?",
            lines=2,
        )

    ask_btn = gr.Button("Poser la question 🔍")

    with gr.Tab("Réponse finale"):
        answer_md = gr.Markdown()

    with gr.Tab("Grounding / Fiabilité"):
        grounding_md = gr.Markdown()

    with gr.Tab("Contexte utilisé"):
        context_md = gr.Markdown()

    ask_btn.click(
        answer_query,
        inputs=[question],
        outputs=[answer_md, grounding_md, context_md],
    )

if __name__ == "__main__":
    demo.launch()
