from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

from llama_index.core.schema import NodeWithScore

from core.index.index_manager import load_vector_index, VectorIndexConfig
from core.knowledge_graph.kg_client import KGClient, KGFact as RawKGFact

logger = logging.getLogger(__name__)



@dataclass
class VectorRetrieverConfig:

    persist_dir: Path = Path("data/vector_store/sec_demo_index")
    top_k: int = 8


@dataclass
class DualRetrieverConfig:
    """
    Config globale du DualRetriever (vectoriel + KG).
    """

    # config pour l'index vectoriel
    vector_index_config: VectorRetrieverConfig = field(
        default_factory=VectorRetrieverConfig
    )

    # nombre de résultats à renvoyer pour chaque voie
    top_k_vector: int = 8
    top_k_kg: int = 8

    # chemin du KG JSON (triples ou faits)
    kg_json_path: Path = Path("data/graph_store/kg_facts.json")
    # seuil minimum d'autorité pour les faits KG
    kg_min_authority_score: float = 0.0


@dataclass
class DualRetrieverResult:
    """
    Résultat combiné du retriever :
      - passages vectoriels (Node With Score)
      - faits KG (dict simplifiés)
    """

    vector_nodes: List[NodeWithScore]
    kg_facts: List[Dict[str, Any]]


# ---------------------------Implémentation du DualRetriever-------------------------------------------

class DualRetriever:
    """
    Orchestration de la récupération double-voie :
      - voie vectorielle (VectorStoreIndex)
      - voie KG 
    """

    def __init__(self, config: Optional[DualRetrieverConfig] = None) -> None:
        self.config = config or DualRetrieverConfig()

        # Index vectoriel
        self._init_vector_index()

        # Client KG
        self.kg_client = KGClient(
            kg_json_path=self.config.kg_json_path,
            min_authority_score=self.config.kg_min_authority_score,
        )

        logger.info(
            "DualRetriever initialisé avec top_k_vector=%d, top_k_kg=%d, persist_dir=%s",
            self.config.top_k_vector,
            self.config.top_k_kg,
            self.config.vector_index_config.persist_dir,
        )

    # -------------------------------Initialisation index vectoriel-------------------------------
  
    def _init_vector_index(self) -> None:
      
        persist_dir = self.config.vector_index_config.persist_dir
        logger.info(
            "Initialisation de l'index vectoriel depuis %s",
            persist_dir,
        )

        # on construit une config pour index_manager
        vec_cfg = VectorIndexConfig(persist_dir=persist_dir)
        self._vector_index = load_vector_index(config=vec_cfg)

        self._vector_retriever = self._vector_index.as_retriever(
            similarity_top_k=self.config.top_k_vector
        )

    # ---------------------------Récupération double-voie pour une query----------------------------------

    def retrieve(self, query: str) -> DualRetrieverResult:
     
        query = query.strip()
        if not query:
            return DualRetrieverResult(vector_nodes=[], kg_facts=[])

        #  Voie vectorielle
        logger.info(
            "Récupération vectorielle pour query=%r (top_k=%d)",
            query,
            self.config.top_k_vector,
        )
        vector_nodes: List[NodeWithScore] = self._vector_retriever.retrieve(query)

        #  Voie KG
        logger.info(
            "Récupération KG pour query=%r (top_k=%d)",
            query,
            self.config.top_k_kg,
        )
        raw_facts: List[RawKGFact] = self.kg_client.search(
            query=query,
            top_k=self.config.top_k_kg,
        )

        kg_facts: List[Dict[str, Any]] = []
        for f in raw_facts:
            meta = getattr(f, "metadata", {}) or {}

            kg_facts.append(
                {
                    "id": getattr(f, "id", None),
                    "subject": getattr(f, "subject", meta.get("subject")),  
                    "relation": getattr(f, "relation", meta.get("relation")), 
                    "object": getattr(f, "object", meta.get("object")),       
                    "score": getattr(f, "score", 0.0),
                    "metadata": meta,  
                }
            )

        logger.info(
            "DualRetriever → %d passages vectoriels, %d faits KG",
            len(vector_nodes),
            len(kg_facts),
        )

        return DualRetrieverResult(
            vector_nodes=vector_nodes,
            kg_facts=kg_facts,
        )

   
    def retrieve_vector_only(self, query: str) -> List[NodeWithScore]:
        query = query.strip()
        if not query:
            return []
        return self._vector_retriever.retrieve(query)



def get_default_dual_retriever(
    persist_dir: str | Path = "data/vector_store/sec_demo_index",
) -> DualRetriever:
  
    cfg = DualRetrieverConfig()
    cfg.vector_index_config.persist_dir = Path(persist_dir)
    return DualRetriever(config=cfg)



if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    retriever = get_default_dual_retriever()

    q = "Quel est le montant de la dette à long terme d'Apple en 2025 ?"
    result = retriever.retrieve(q)

    print(f"\n=== VECTOR NODES ({len(result.vector_nodes)}) ===")
    for i, n in enumerate(result.vector_nodes[:3]):
        print(f"[{i}] score={n.score:.4f} source={n.node.metadata.get('source')}")

    print(f"\n=== KG FACTS ({len(result.kg_facts)}) ===")
    for i, f in enumerate(result.kg_facts[:5]):
        print(f"[{i}] score={f['score']:.4f} text={f['text'][:120]}...")
