from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

# LLM gratuit : OLLAMA
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from llama_index.core.schema import NodeWithScore


# from core.retrieval.query_transformer import (
#     RAGFusionQueryTransformer,
#     QueryTransformerConfig,
# )

from core.retrieval.dual_retriever import (
    DualRetrieverConfig,
    get_default_dual_retriever,
)
from core.retrieval.reranker_trust import (
    TrustRerankerConfig,
    rerank_nodes_with_trust,
)
from core.generation.generator import (
    TrustRAGGenerator,
    GeneratorConfig,
)
from core.generation.grounding_guardrails import (
    GroundingGuardrails,
    GroundingGuardrailsConfig,
    GroundingCheckResult,
)


logger = logging.getLogger(__name__)


Settings.llm = Ollama(model="mistral")
logger.info("Settings.llm forcé sur Ollama mistral)")


# CONFIG DU PIPELINE

@dataclass
class RetrievalPipelineConfig:
    """Config globale du pipeline trustRAG."""

    vector_index_dir: Path = Path("data/vector_store/sec_demo_index")

    dual_retriever: DualRetrieverConfig = field(
        default_factory=DualRetrieverConfig
    )

    # query_transformer: QueryTransformerConfig = field(
    #     default_factory=QueryTransformerConfig
    # )

    reranker: TrustRerankerConfig = field(
        default_factory=lambda: TrustRerankerConfig(
            w_similarity=0.6,
            w_authority=0.4,
            min_authority_score=None,
        )
    )
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)

    guardrails: GroundingGuardrailsConfig = field(
        default_factory=lambda: GroundingGuardrailsConfig(
            min_authority_score_for_answer=0.5,
            min_lexical_overlap=0.15,
        )
    )


# PIPELINE RETRIEVAL + GENERATION

class RetrievalPipeline:
    def __init__(self, config: Optional[RetrievalPipelineConfig] = None) -> None:
        self.config = config or RetrievalPipelineConfig()

        # configure le chemin de l’index vectoriel
        self.config.dual_retriever.vector_index_config.persist_dir = (
            self.config.vector_index_dir
        )

        # instanciation des composants
        self.dual_retriever = get_default_dual_retriever(
            persist_dir=self.config.vector_index_dir
        )

        # self.query_transformer = RAGFusionQueryTransformer(
        #     config=self.config.query_transformer,
        #     dual_retriever=self.dual_retriever,
        # )

        self.generator = TrustRAGGenerator(config=self.config.generator)
        self.guardrails = GroundingGuardrails(config=self.config.guardrails)

        logger.info(
            "RetrievalPipeline initialisé avec index=%s",
            self.config.vector_index_dir,
        )

 
    def answer_question(self, question: str) -> Dict[str, Any]:
        question = question.strip()
        if not question:
            return {
                "final_answer": "Question vide.",
                "candidate_answer": "",
                "grounding_check": None,
                "vector_nodes": [],
                "kg_facts": [],
            }

        logger.info("=== trustRAG pipeline : nouvelle requête ===")
        logger.info("Question : %s", question)

        dual_result = self.dual_retriever.retrieve(question)
        vector_nodes = dual_result.vector_nodes
        kg_facts = dual_result.kg_facts

        #  Reranking utilisant le score de confiance
        ranked_nodes = rerank_nodes_with_trust(
            vector_nodes,
            config=self.config.reranker,
        )

        # Génération
        candidate_answer, ctx_summary = self.generator.generate_answer(
            question=question,
            context_nodes=ranked_nodes,
            kg_facts=kg_facts,
            temperature=0.2,
        )

        #  Vérification de grounding
        final_answer, grounding_check = self.guardrails.decide_answer(
            answer=candidate_answer,
            context_nodes=ranked_nodes,
            kg_facts=kg_facts,
        )

        return {
            "final_answer": final_answer,
            "candidate_answer": candidate_answer,
            "grounding_check": grounding_check,
            "vector_nodes": ranked_nodes,
            "kg_facts": kg_facts,
            "used_context_summary": ctx_summary,
        }



def get_default_retrieval_pipeline() -> RetrievalPipeline:
    return RetrievalPipeline()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pipeline = get_default_retrieval_pipeline()

    question = "Quel est le montant de la dette à long terme d'Apple en 2025 ?"
    result = pipeline.answer_question(question)

    print("\n=== RÉPONSE FINALE ===")
    print(result["final_answer"])

    print("\n=== CHECK GROUNDING ===")
    print(result["grounding_check"])

    print("\n=== EXTRAIT CONTEXTE UTILISÉ ===")
    print(result["used_context_summary"][:1000])
