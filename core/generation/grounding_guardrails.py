from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from llama_index.core.schema import NodeWithScore

logger = logging.getLogger(__name__)


# -----------------------Dataclass de résultat-----------------------------------------------

@dataclass
class GroundingCheckResult:
  

    is_grounded: bool
    reasons: List[str]
    max_authority_score: float
    lexical_overlap: float


# -------------------------------Config des guardrails ---------------------------------------

@dataclass
class GroundingGuardrailsConfig:


    min_authority_score_for_answer: float = 0.3
    min_lexical_overlap: float = 0.05
    max_context_nodes_for_overlap: int = 8
    refusal_message: str = (
        "Je ne peux pas répondre de façon fiable à cette question avec les "
        "sources de confiance actuellement disponibles. "
        "Je préfère m'abstenir plutôt que d'halluciner."
    )


# ----------------------------Guardrails------------------------------------------

class GroundingGuardrails:
    def __init__(self, config: Optional[GroundingGuardrailsConfig] = None) -> None:
        self.config = config or GroundingGuardrailsConfig()
        logger.info(
            "GroundingGuardrails init (min_auth=%.2f, min_overlap=%.2f)",
            self.config.min_authority_score_for_answer,
            self.config.min_lexical_overlap,
        )

    # -----------------------Extraction du meilleur score V4-A du contexte-------------------------------------------
  
    @staticmethod
    def _get_authority_from_metadata(md: Dict[str, Any]) -> float:
        for key in [
            "source_authority_score_base",
            "authority_score_base",
            "v4_authority_score",
        ]:
            if key in md:
                try:
                    return float(md[key])
                except Exception:
                    continue
        return 0.3 

    def _compute_context_authority(
        self,
        nodes: List[NodeWithScore],
        kg_facts: List[Dict[str, Any]],
    ) -> float:
        scores: List[float] = []

        for node in nodes:
            md = node.node.metadata or {}
            scores.append(self._get_authority_from_metadata(md))

        for fact in kg_facts:
            scores.append(float(fact.get("source_authority_score_base", 0.3)))

        return max(scores) if scores else 0.0

    # ----------------------------------Recouvrement lexical réponse ↔ contexte--------------------------------
    
    def _compute_lexical_overlap(
        self,
        answer: str,
        nodes: List[NodeWithScore],
        kg_facts: List[Dict[str, Any]],
    ) -> float:
        
        answer = answer.strip().lower()
        if not answer:
            return 0.0

        answer_tokens = set(re.findall(r"\w+", answer))

        context_parts: List[str] = []

        # concat texte de quelques nodes
        for node in nodes[: self.config.max_context_nodes_for_overlap]:
            context_parts.append(node.node.get_content() or "")

        # représentation textuelle des faits KG
        for fact in kg_facts:
            try:
                context_parts.append(" ".join(f"{k}: {v}" for k, v in fact.items()))
            except Exception:
                continue

        context_text = " ".join(context_parts).lower()
        context_tokens = set(re.findall(r"\w+", context_text))

        if not context_tokens:
            return 0.0

        inter = answer_tokens.intersection(context_tokens)
        overlap = len(inter) / max(len(answer_tokens), 1)

        return overlap

    # ---------------------------Vérification principale---------------------------------------
 
    def evaluate_answer(
        self,
        answer: str,
        context_nodes: List[NodeWithScore],
        kg_facts: List[Dict[str, Any]],
    ) -> GroundingCheckResult:

        reasons: List[str] = []

        if not context_nodes and not kg_facts:
            reasons.append("Aucun contexte fourni (nodes ni KG).")
            return GroundingCheckResult(
                is_grounded=False,
                reasons=reasons,
                max_authority_score=0.0,
                lexical_overlap=0.0,
            )

        max_auth = self._compute_context_authority(context_nodes, kg_facts)
        overlap = self._compute_lexical_overlap(answer, context_nodes, kg_facts)

        fail_auth = max_auth < self.config.min_authority_score_for_answer
        fail_overlap = overlap < self.config.min_lexical_overlap

        if fail_auth and fail_overlap:
            # Les deux conditions sont mauvaises → on refuse
            if fail_auth:
                reasons.append(
                    f"Score d'autorité max du contexte trop faible : {max_auth:.2f} "
                    f"(min requis = {self.config.min_authority_score_for_answer:.2f})."
                )
            if fail_overlap:
                reasons.append(
                    f"Recouvrement lexical trop faible entre la réponse et le contexte : "
                    f"{overlap:.2f} (min requis = {self.config.min_lexical_overlap:.2f})."
                )
            is_grounded = False
        else:
            # Au moins un des deux (autorité / overlap) est satisfaisant → on accepte
            is_grounded = True
            reasons.append(
                f"Réponse considérée comme suffisamment ancrée "
                f"(max_auth={max_auth:.2f}, overlap={overlap:.2f})."
            )

        logger.info(
            "GroundingCheckResult(is_grounded=%s, max_auth=%.2f, overlap=%.2f)",
            is_grounded,
            max_auth,
            overlap,
        )

        return GroundingCheckResult(
            is_grounded=is_grounded,
            reasons=reasons,
            max_authority_score=max_auth,
            lexical_overlap=overlap,
        )

    # --------------------------API “answer or refuse”----------------------------------------
    
    def decide_answer(
        self,
        answer: str,
        context_nodes: List[NodeWithScore],
        kg_facts: List[Dict[str, Any]],
    ) -> Tuple[str, GroundingCheckResult]:
        check = self.evaluate_answer(answer, context_nodes, kg_facts)
        if check.is_grounded:
            return answer, check

        refusal = self.config.refusal_message
        return refusal, check



def get_default_grounding_guardrails() -> GroundingGuardrails:
    return GroundingGuardrails()
