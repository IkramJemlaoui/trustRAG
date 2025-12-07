from __future__ import annotations

import logging
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama  

# si le chemin diffère chez toi, adapte l'import ci-dessous
from core.knowledge_graph.kg_client import KGFact

logger = logging.getLogger(__name__)


@dataclass
class GeneratorConfig:
    # modèle Ollama utilisé
    model_name: str = "qwen2.5:0.5b-instruct"
    # timeout HTTP en secondes
    request_timeout: float = 1000.0

    # limites de contexte
    max_context_nodes: int = 4
    max_context_chars: int = 2500
    max_kg_facts: int = 8

    system_prompt: str = textwrap.dedent(
        """
        Tu es un assistant financier fiable connecté au système trustRAG.
        Tu DOIS te baser UNIQUEMENT sur le contexte fourni.
        Si une information n'est pas présente dans le contexte, dis-le honnêtement.
        Réponds dans la langue de la question (français ou anglais).
        """
    ).strip()


class TrustRAGGenerator:
    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        llm: Optional[LLM] = None,
    ) -> None:
        self.config = config or GeneratorConfig()

        if llm is not None:
            self.llm = llm
        else:
            self.llm = Ollama(
                model=self.config.model_name,
                request_timeout=self.config.request_timeout,
            )

        logger.info(
            "TrustRAGGenerator initialisé (model=%s, max_nodes=%d, max_chars=%d, max_kg_facts=%d)",
            getattr(self.llm, "model", self.config.model_name),
            self.config.max_context_nodes,
            self.config.max_context_chars,
            self.config.max_kg_facts,
        )

    # --------------------- Construction du texte de contexte ---------------------
    def _build_context_text(
        self,
        nodes: List[NodeWithScore],
        kg_facts: List[Any],
    ) -> str:
        node_texts: List[str] = []
        total_chars = 0

        # on applique les limites ici aussi
        nodes = nodes[: self.config.max_context_nodes]
        kg_facts = kg_facts[: self.config.max_kg_facts]

        # --------- contexte textuel (nodes) ----------
        for i, node in enumerate(nodes):
            content = node.node.get_content() or ""
            content = content.strip()
            if not content:
                continue

            if total_chars >= self.config.max_context_chars:
                break

            remaining = self.config.max_context_chars - total_chars
            snippet = content[:remaining]

            md = node.node.metadata or {}
            src = md.get("source") or md.get("source_system") or "unknown_source"

            wrapped = textwrap.indent(snippet, prefix="  ")
            node_block = f"[DOC #{i} | source={src} | score={node.score:.4f}]\n{wrapped}\n"
            node_texts.append(node_block)
            total_chars += len(snippet)

        # --------- faits structurés (KG) ----------
        fact_lines: List[str] = []
        for i, fact in enumerate(kg_facts):
            try:
                if isinstance(fact, dict):
                    items = ", ".join(f"{k}={v}" for k, v in fact.items())
                elif isinstance(fact, KGFact):
                    items = (
                        f"subject={fact.subject}, relation={fact.relation}, "
                        f"object={fact.object}, score={fact.score}, "
                        f"meta={fact.metadata}"
                    )
                else:
                    items = str(fact)
            except Exception:
                items = str(fact)

            fact_lines.append(f"- FACT #{i}: {items}")

        parts: List[str] = []

        if node_texts:
            parts.append("## Contexte textuel (documents récupérés)\n")
            parts.extend(node_texts)

        if fact_lines:
            parts.append("\n## Faits structurés (Knowledge Graph)\n")
            parts.extend(fact_lines)

        if not parts:
            return "Aucun contexte fiable n'a été fourni."

        return "\n".join(parts)

    # ----------------------------- Messages pour le LLM -----------------------------
    def _build_messages(
        self,
        question: str,
        nodes: List[NodeWithScore],
        kg_facts: List[Any],
    ) -> List[ChatMessage]:
        context_text = self._build_context_text(nodes, kg_facts)

        user_prompt = textwrap.dedent(
            f"""
            Question utilisateur :
            {question}

            Contexte (à utiliser exclusivement) :
            {context_text}

            Tâche :
            - Donne une réponse structurée, claire et concise.
            - Ne fabrique aucun chiffre qui n'est pas présent dans le contexte.
            - Si la réponse exacte n'est pas disponible, explique ce qui manque.
            - Si la question est de type "Apple parle-t-elle de...", "Est-ce que...",
              commence ta réponse par "Oui, ..." ou "Non, ...", puis justifie en citant le contexte.
            """
        ).strip()

        return [
            ChatMessage(role=MessageRole.SYSTEM, content=self.config.system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]

    # ----------------------------- Appel principal -----------------------------
    def generate_answer(
        self,
        question: str,
        context_nodes: List[NodeWithScore],
        kg_facts: List[Any],
        temperature: float = 0.0,
    ) -> Tuple[str, str]:
        # on tronque AVANT de construire les messages
        context_nodes = (context_nodes or [])[: self.config.max_context_nodes]
        kg_facts = (kg_facts or [])[: self.config.max_kg_facts]

        messages = self._build_messages(question, context_nodes, kg_facts)

        logger.info(
            "Appel LLM (Ollama) pour génération de réponse (nodes=%d, facts=%d).",
            len(context_nodes),
            len(kg_facts),
        )

        try:
            resp = self.llm.chat(messages, temperature=temperature)
            answer_text = (resp.message.content or "").strip()
        except Exception as e:
            logger.error("Erreur lors de l'appel LLM (generation) : %s", e)
            answer_text = (
                "Je n'ai pas pu générer de réponse car le modèle local a mis trop de temps "
                "ou a rencontré une erreur. Essaie avec une question plus courte, "
                "ou relance le service Ollama."
            )

        # résumé du contexte utilisé (pour affichage dans la GUI)
        used_context_summary = self._build_context_text(
            context_nodes[: min(len(context_nodes), 3)],
            kg_facts[: min(len(kg_facts), 3)],
        )

        return answer_text, used_context_summary


def get_default_generator() -> TrustRAGGenerator:
    return TrustRAGGenerator()
