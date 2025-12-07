from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 0. Config du reranker de confiance
# ----------------------------------------------------------------------
@dataclass
class TrustRerankerConfig:
    """
    Reranking basé sur :
      - score de similarité (node.score)
      - V4-A : source_authority_score_base (métadonnée trustRAG)

    final_score = w_sim * sim + w_auth * authority

    Si min_authority_score est défini :
      - les nodes en dessous sont soit pénalisés, soit filtrés.
    """

    w_similarity: float = 0.7
    w_authority: float = 0.3
    min_authority_score: Optional[float] = None  # ex: 0.5 pour filtrer < officiel


# ----------------------------------------------------------------------
# 1. Reranker trustRAG
# ----------------------------------------------------------------------
class TrustReranker:
    def __init__(self, config: Optional[TrustRerankerConfig] = None) -> None:
        self.config = config or TrustRerankerConfig()
        logger.info(
            "Initialisation TrustReranker(w_similarity=%.2f, w_authority=%.2f, min_auth=%s)",
            self.config.w_similarity,
            self.config.w_authority,
            str(self.config.min_authority_score),
        )

    # ------------------------------------------------------------------
    # 1.1 Calcul d'un score final pour un node
    # ------------------------------------------------------------------
    def _get_authority_score(self, node: NodeWithScore) -> float:
        """
        Récupère V4-A depuis les métadonnées du node.
        Fallback à 0.3 (public_general) si absent.
        """
        md = node.node.metadata or {}
        # on accepte plusieurs variantes par sécurité
        for key in [
            "source_authority_score_base",
            "authority_score_base",
            "v4_authority_score",
        ]:
            if key in md:
                try:
                    return float(md[key])
                except Exception:
                    logger.debug("Impossible de caster %s en float pour %s", key, md[key])
        return 0.3  # fallback : public_general

    def _compute_final_score(self, node: NodeWithScore) -> float:
        sim = float(node.score or 0.0)
        auth = self._get_authority_score(node)

        # normalisation simple : sim est souvent entre 0 et 1, auth déjà dans [0,1]
        final_score = (
            self.config.w_similarity * sim
            + self.config.w_authority * auth
        )
        return final_score

    # ------------------------------------------------------------------
    # 1.2 Reranking d'une liste de nodes
    # ------------------------------------------------------------------
    def rerank(self, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Applique le reranking aux nodes :

          1. Optionnel : filtre ceux en dessous de min_authority_score.
          2. Trie par score final décroissant.

        Le score original (similarité) reste accessible dans node.score,
        mais on ajoute 'trust_final_score' et 'trust_authority_score'
        dans node.node.metadata pour traçabilité.
        """
        if not nodes:
            return []

        filtered: List[NodeWithScore] = []
        for node in nodes:
            auth = self._get_authority_score(node)
            if (
                self.config.min_authority_score is not None
                and auth < self.config.min_authority_score
            ):
                logger.debug(
                    "Node filtré (authority=%.2f < min_authority_score=%.2f)",
                    auth,
                    self.config.min_authority_score,
                )
                continue

            final_score = self._compute_final_score(node)

            # on injecte les scores dans les métadonnées pour audit
            md = node.node.metadata or {}
            md = dict(md)  # copy
            md["trust_authority_score"] = auth
            md["trust_similarity_score"] = float(node.score or 0.0)
            md["trust_final_score"] = final_score
            node.node.metadata = md

            # on peut aussi surcharger node.score par le score final,
            # ce qui permettra de réutiliser plus facilement dans LlamaIndex.
            node.score = final_score

            filtered.append(node)

        # tri final par score décroissant
        filtered.sort(key=lambda n: n.score or 0.0, reverse=True)

        logger.info(
            "Reranking terminé : %d nodes en entrée → %d nodes en sortie",
            len(nodes),
            len(filtered),
        )

        return filtered


# ----------------------------------------------------------------------
# 2. Helper fonctionnel
# ----------------------------------------------------------------------
def rerank_nodes_with_trust(
    nodes: List[NodeWithScore],
    config: Optional[TrustRerankerConfig] = None,
) -> List[NodeWithScore]:
    """
    Helper simple pour le pipeline :
      - prend une liste de nodes (DualRetriever.vector_nodes)
      - renvoie la liste rerankée
    """
    reranker = TrustReranker(config=config)
    return reranker.rerank(nodes)


# ----------------------------------------------------------------------
# 3. Petit main de test (optionnel)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    from core.retrieval.dual_retriever import get_default_dual_retriever
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)

    persist_dir = Path("data/vector_store/sec_demo_index")
    dual = get_default_dual_retriever(persist_dir=persist_dir)

    query = "What is Apple long-term debt?"
    result = dual.retrieve(query)

    print(f"Avant reranking :")
    for i, n in enumerate(result.vector_nodes[:3]):
        print(f"- #{i} score={n.score:.4f} source={n.node.metadata.get('source')}")

    reranked = rerank_nodes_with_trust(
        result.vector_nodes,
        config=TrustRerankerConfig(
            w_similarity=0.6,
            w_authority=0.4,
            min_authority_score=None,
        ),
    )

    print(f"\nAprès reranking :")
    for i, n in enumerate(reranked[:3]):
        print(
            f"- #{i} final={n.score:.4f} "
            f"(sim={n.node.metadata.get('trust_similarity_score'):.4f}, "
            f"auth={n.node.metadata.get('trust_authority_score')}) "
            f"source={n.node.metadata.get('source')}"
        )
