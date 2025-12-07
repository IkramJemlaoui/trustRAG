from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from llama_index.core import Settings
from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore

from core.retrieval.dual_retriever import (
    DualRetriever,
    DualRetrieverConfig,
    DualRetrieverResult,   # 👈 nom corrigé
    get_default_dual_retriever,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 0. Config & résultat RAG Fusion
# ----------------------------------------------------------------------
@dataclass
class QueryTransformerConfig:
    """
    Config pour le module de RAG Fusion :
      - enabled : activer/désactiver la génération de variantes
      - num_variants : nombre de reformulations à générer (question incluse)
      - temperature : température du LLM pour la génération
    """

    enabled: bool = True
    num_variants: int = 4
    temperature: float = 0.3


@dataclass
class RAGFusionQueryResult:
    """
    Résultat du QueryTransformer avec RAG Fusion :
      - vector_nodes : passages récupérés via la voie vectorielle
      - kg_facts : faits récupérés via la voie KG
      - all_queries : liste (question originale + variantes)
    """

    vector_nodes: List[NodeWithScore]
    kg_facts: List[Dict[str, Any]]
    all_queries: List[str]


# ----------------------------------------------------------------------
# 1. Implémentation RAG Fusion
# ----------------------------------------------------------------------
class RAGFusionQueryTransformer:
    """
    Utilise un LLM (Settings.llm ou fourni) pour générer plusieurs variantes
    de la requête utilisateur, puis appelle le DualRetriever sur chacune.

    Logique :
      1) Génère N requêtes (question + reformulations)
      2) Appelle DualRetriever.retrieve(query) pour chaque requête
      3) Fusionne les résultats (déduplication + agrégation)
    """

    def __init__(
        self,
        config: Optional[QueryTransformerConfig] = None,
        dual_retriever: Optional[DualRetriever] = None,
        llm: Optional[LLM] = None,
    ) -> None:
        self.config = config or QueryTransformerConfig()
        self.dual_retriever = dual_retriever or get_default_dual_retriever()

        # LLM utilisé pour générer les variantes
        self.llm: Optional[LLM] = llm or Settings.llm
        if self.llm is None and self.config.enabled:
            logger.warning(
                "RAGFusionQueryTransformer : aucun LLM configuré (Settings.llm est None). "
                "La génération de variantes sera désactivée."
            )
            self.config.enabled = False

        logger.info(
            "RAGFusionQueryTransformer initialisé (num_variants=%d, temp=%.2f)",
            self.config.num_variants,
            self.config.temperature,
        )

    # --------------------------------------------------------------
    # 1.1 Génération de variantes de requêtes
    # --------------------------------------------------------------
    def _generate_variants(self, question: str) -> List[str]:
        """
        Génère une liste de variantes de la question en utilisant un LLM.
        La question originale est toujours incluse en première position.
        """
        question = question.strip()
        if not question:
            return []

        # fallback : pas de variantes si désactivé ou pas de LLM
        if not self.config.enabled or self.llm is None:
            return [question]

        system_prompt = (
            "Tu es un assistant spécialisé en recherche d'information financière. "
            "Pour une question utilisateur donnée, tu dois proposer plusieurs "
            "reformulations naturelles qui préservent le même sens, pour "
            "améliorer la recherche dans une base de connaissances.\n\n"
            "Contraintes :\n"
            "- 1 reformulation par ligne\n"
            "- Pas de numérotation\n"
            "- Pas de commentaire autour, uniquement les variantes\n"
        )

        user_prompt = (
            f"Question utilisateur :\n{question}\n\n"
            f"Génère {self.config.num_variants - 1} reformulations différentes."
        )

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
            ChatMessage(role=MessageRole.USER, content=user_prompt),
        ]

        try:
            resp = self.llm.chat(messages=messages, temperature=self.config.temperature)
            raw_text = resp.message.content or ""
            lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
            # on prend au max num_variants - 1 lignes
            variants = lines[: max(0, self.config.num_variants - 1)]
        except Exception as e:
            logger.exception(
                "Erreur lors de la génération de variantes de requête : %s", e
            )
            variants = []

        # on s'assure que la question est en première position
        all_q = [question]
        for v in variants:
            if v not in all_q:
                all_q.append(v)

        return all_q

    # --------------------------------------------------------------
    # 1.2 RAG Fusion : retrieve_with_fusion
    # --------------------------------------------------------------
    def retrieve_with_fusion(self, question: str) -> RAGFusionQueryResult:
        """
        Pipeline complet :
          1) Génère des variantes de la question
          2) Appelle DualRetriever pour chaque variante
          3) Fusionne les résultats avec déduplication
        """
        question = question.strip()
        if not question:
            return RAGFusionQueryResult(vector_nodes=[], kg_facts=[], all_queries=[])

        # 1) Génération des variantes
        all_queries = self._generate_variants(question)
        logger.info("Requêtes RAG Fusion : %s", all_queries)

        # 2) Appels au DualRetriever
        all_vector_nodes: List[NodeWithScore] = []
        all_kg_facts: List[Dict[str, Any]] = []

        for q in all_queries:
            dr_result: DualRetrieverResult = self.dual_retriever.retrieve(q)
            all_vector_nodes.extend(dr_result.vector_nodes)
            all_kg_facts.extend(dr_result.kg_facts)

        # 3) Fusion / déduplication des résultats
        dedup_vector = self._deduplicate_nodes(all_vector_nodes)
        dedup_kg = self._deduplicate_kg_facts(all_kg_facts)

        logger.info(
            "RAG Fusion fini pour %r → %d nodes, %d facts",
            question,
            len(dedup_vector),
            len(dedup_kg),
        )

        return RAGFusionQueryResult(
            vector_nodes=dedup_vector,
            kg_facts=dedup_kg,
            all_queries=all_queries,
        )

    # --------------------------------------------------------------
    # 1.3 Déduplication nodes & facts
    # --------------------------------------------------------------
    @staticmethod
    def _deduplicate_nodes(nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        seen_ids = set()
        dedup: List[NodeWithScore] = []
        for n in nodes:
            node_id = n.node.node_id
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            dedup.append(n)
        return dedup

    @staticmethod
    def _deduplicate_kg_facts(
        facts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        seen_ids = set()
        dedup: List[Dict[str, Any]] = []
        for f in facts:
            fid = f.get("id")
            if fid is not None and fid in seen_ids:
                continue
            if fid is not None:
                seen_ids.add(fid)
            dedup.append(f)
        return dedup


# ----------------------------------------------------------------------
# 2. Helper global
# ----------------------------------------------------------------------
def get_default_query_transformer(
    dual_retriever: Optional[DualRetriever] = None,
) -> RAGFusionQueryTransformer:
    return RAGFusionQueryTransformer(
        config=QueryTransformerConfig(),
        dual_retriever=dual_retriever or get_default_dual_retriever(),
    )


# ----------------------------------------------------------------------
# 3. Petit main de test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    qt = get_default_query_transformer()
    q = "Quel est le montant de la dette à long terme d'Apple en 2025 ?"
    res = qt.retrieve_with_fusion(q)

    print("Queries utilisées :", res.all_queries)
    print("Vector nodes :", len(res.vector_nodes))
    print("KG facts :", len(res.kg_facts))
