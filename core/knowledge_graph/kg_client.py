from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class KGFact:
    id: str
    subject: str
    relation: str
    object: Any
    score: float
    metadata: Dict[str, Any]


class KGClient:
    def __init__(
        self,
        kg_json_path: str | Path = "data/graph_store/kg_facts.json",
        min_authority_score: float = 0.0,
    ):
        self.kg_json_path = Path(kg_json_path)

        if not self.kg_json_path.exists():
            logger.warning("KG file missing: %s", kg_json_path)
            self.triples = []
            return

        raw = json.loads(self.kg_json_path.read_text(encoding="utf-8"))

        self.triples = [
            t for t in raw if float(t.get("authority_score", 0)) >= min_authority_score
        ]

        logger.info(
            "KGClient chargé : %d triples (authority ≥ %.2f)",
            len(self.triples),
            min_authority_score,
        )

    def search(self, query: str, top_k=8) -> List[KGFact]:
        if not query.strip():
            return []

        q = set(query.lower().split())
        results = []

        for t in self.triples:
            txt = f"{t['subject']} {t['relation']} {t['object']}".lower()
            tokens = set(txt.split())

            overlap = len(tokens & q)
            if overlap == 0:
                continue

            score = overlap * (1 + float(t["authority_score"]))

            results.append(
                KGFact(
                    id=t["id"],
                    subject=t["subject"],
                    relation=t["relation"],
                    object=t["object"],
                    score=score,
                    metadata=t,
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
