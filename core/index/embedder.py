from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from llama_index.core import Document
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)


# --------------------------Config de l'embedder--------------------------------------------

@dataclass
class TrustRAGEmbedderConfig:
    # Modèle HuggingFace local 
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 32


# --------------------------Wrapper TrustRAG autour d'un BaseEmbedding local--------------------------------------------

class TrustRAGEmbedder:


    def __init__(self, config: Optional[TrustRAGEmbedderConfig] = None) -> None:
        self.config = config or TrustRAGEmbedderConfig()

        logger.info(
            "Initialisation TrustRAGEmbedder (HuggingFace) avec model_name=%s",
            self.config.model_name,
        )

        # Modèle HF local 
        self._model: BaseEmbedding = HuggingFaceEmbedding(
            model_name=self.config.model_name
        )

    @property
    def model(self) -> BaseEmbedding:
        """Retourne l'objet BaseEmbedding utilisé par LlamaIndex."""
        return self._model

    def embed_query(self, query: str) -> List[float]:
        return self._model.get_text_embedding(query)

    def embed_documents(
        self,
        documents: Sequence[Document],
    ) -> List[List[float]]:
        texts = [doc.text for doc in documents]
        if not texts:
            return []
        return self._model.get_text_embedding_batch(texts)

    def embed_documents_with_docs(
        self,
        documents: Sequence[Document],
    ) -> List[Tuple[Document, List[float]]]:
        embs = self.embed_documents(documents)
        return list(zip(documents, embs))


# -------------------------Helper global pour index_manager---------------------------------------------

def get_default_embedder() -> TrustRAGEmbedder:
   
    return TrustRAGEmbedder()
