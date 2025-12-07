from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

from llama_index.core import Document

logger = logging.getLogger(__name__)


# ------------------------------structure parent / enfant----------------------------------------

@dataclass
class ParentChildResult:
    parent_docs: List[Document]
    child_chunks: List[Document]


# --------------------------V5 - Détermination du type de contenu (routing)--------------------------------------------

def infer_content_structure_type(metadata: Dict) -> str:

    source = (metadata.get("source") or "").lower()
    source_system = (metadata.get("source_system") or "").lower()

    if "sec_edgar" in source or "sec" in source_system:
        return "unstructured_narrative"

    if "financial_news_events" in source or "financial_impact_ai_lab" in source_system:
        return "unstructured_narrative"

    return "unstructured_narrative"


# ------------------------------------------------Chunking simple avec overlap :Parent Document ----------------------

def _chunk_text_with_overlap(
    text: str,
    max_chars: int = 1000,
    overlap_chars: int = 200,
) -> List[str]:

    if not text:
        return []

    text = text.strip()
    n = len(text)

    if n <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = end - overlap_chars

    return chunks


# ---------------------------- Construction Parent / Enfants pour le RAG ------------------------------------------

def build_parent_child_chunks(
    documents: List[Document],
    max_chars: int = 1000,
    overlap_chars: int = 200,
) -> ParentChildResult:
    """
    Prend une liste de `Document` (SEC filings, news, etc.)
    et renvoie :
      - parent_docs : Documents "parents" (gros contexte)
      - child_chunks : petits chunks indexables (vector store)

    Chaque chunk enfant contient :
      - parent_doc_id : identifiant du parent
      - chunk_id      : identifiant unique du chunk
      - chunk_index   : position séquentielle
      - content_structure_type (V5)
    """
    parent_docs: List[Document] = []
    child_chunks: List[Document] = []

    for i, doc in enumerate(documents):
        base_metadata = dict(doc.metadata or {})
        parent_id = base_metadata.get("parent_doc_id") or base_metadata.get("doc_id")

        if not parent_id:
            # on génère un ID stable pour ce parent
            parent_id = str(uuid.uuid4())

        # on marque le parent
        parent_metadata = {
            **base_metadata,
            "doc_id": parent_id,
            "parent_doc_id": parent_id,
            "chunk_role": "parent",
            "content_structure_type": infer_content_structure_type(base_metadata),
        }

        parent_doc = Document(
            text=doc.text,
            metadata=parent_metadata,
        )
        parent_docs.append(parent_doc)

        # on génère les enfants à partir du texte du parent
        chunks = _chunk_text_with_overlap(
            parent_doc.text,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )

        for idx, chunk_text in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            child_metadata = {
                **base_metadata,
                "doc_id": chunk_id,
                "parent_doc_id": parent_id,
                "chunk_role": "child",
                "chunk_index": idx,
                "content_structure_type": infer_content_structure_type(base_metadata),
            }

            child_doc = Document(
                text=chunk_text,
                metadata=child_metadata,
            )
            child_chunks.append(child_doc)

        logger.debug(
            "Document %d → parent_id=%s, nb_chunks_enfants=%d",
            i,
            parent_id,
            len(chunks),
        )

    logger.info(
        "Chunker avancé terminé : %d parents, %d enfants",
        len(parent_docs),
        len(child_chunks),
    )

    return ParentChildResult(parent_docs=parent_docs, child_chunks=child_chunks)


# ----------------------------------interface pour le pipeline ------------------------------------

def chunk_documents_for_vector_index(
    documents: List[Document],
    max_chars: int = 1000,
    overlap_chars: int = 200,
) -> List[Document]:
    """
    Interface pratique pour le pipeline d'indexation :
      - prend les Documents bruts (loader)
      - renvoie uniquement les chunks ENFANTS (petits)
        pour alimenter le vector store.

    Les parents peuvent être stockés dans un autre index
    ou dans un store séparé pour le Parent Document Retriever.
    """
    result = build_parent_child_chunks(
        documents=documents,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )
    return result.child_chunks
