from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings,
)

from .embedder import TrustRAGEmbedder, get_default_embedder

logger = logging.getLogger(__name__)


# ------------------------Configuration de l'index vectoriel----------------------------------------------

@dataclass
class VectorIndexConfig:

    persist_dir: Path = Path("data/vector_store/trustrag_default")


# ----------------------------------Construction d'un nouvel index vectoriel------------------------------------

def build_vector_index(
    documents: List[Document],
    config: Optional[VectorIndexConfig] = None,
    embedder: Optional[TrustRAGEmbedder] = None,
) -> VectorStoreIndex:
 
    if config is None:
        config = VectorIndexConfig()

    if embedder is None:
        embedder = get_default_embedder()

    # branche le modèle d'embedding dans les Settings globaux LlamaIndex
    Settings.embed_model = embedder.model

    persist_dir = config.persist_dir
    persist_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Construction d'un VectorStoreIndex (%d documents) avec persist_dir=%s",
        len(documents),
        persist_dir,
    )

    storage_context = StorageContext.from_defaults()
    
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    logger.info("Persistance de l'index vectoriel dans %s", persist_dir)
    index.storage_context.persist(persist_dir=str(persist_dir))

    return index


# ----------------------Chargement d'un index vectoriel existant------------------------------------------

def load_vector_index(
    config: Optional[VectorIndexConfig] = None,
    embedder: Optional[TrustRAGEmbedder] = None,
) -> VectorStoreIndex:

    if config is None:
        config = VectorIndexConfig()

    if embedder is None:
        embedder = get_default_embedder()

    Settings.embed_model = embedder.model

    persist_dir = config.persist_dir
    if not persist_dir.exists():
        raise FileNotFoundError(
            f"Impossible de charger l'index vectoriel : {persist_dir} introuvable."
        )

    logger.info("Chargement de l'index vectoriel depuis %s", persist_dir)

    storage_context = StorageContext.from_defaults(persist_dir=str(persist_dir))
    index = load_index_from_storage(storage_context=storage_context)

    return index


# ----------------------------- ajout de nouveaux documents---------------------------------------

def update_vector_index_with_documents(
    documents: List[Document],
    config: Optional[VectorIndexConfig] = None,
    embedder: Optional[TrustRAGEmbedder] = None,
) -> VectorStoreIndex:

    if config is None:
        config = VectorIndexConfig()

    if embedder is None:
        embedder = get_default_embedder()

    # charge l'index existant 
    index = load_vector_index(config=config, embedder=embedder)

    logger.info(
        "Ajout de %d nouveaux documents dans l'index vectoriel...",
        len(documents),
    )

    index.insert_documents(documents)
    index.storage_context.persist(persist_dir=str(config.persist_dir))

    return index


# ----------------------------------à implémenter plus tard------------------------------------

def build_kg_index(*args, **kwargs):

    raise NotImplementedError("build_kg_index() sera implémenté dans le module KG.")


def load_kg_index(*args, **kwargs):
  
    raise NotImplementedError("load_kg_index() sera implémenté dans le module KG.")


if __name__ == "__main__":
    import logging
    from core.ingestion.loaders import load_raw_documents_from_sec
    from core.ingestion.chunker_advanced import build_parent_child_chunks

    logging.basicConfig(level=logging.INFO)

    # Exemple : construire un index vectoriel à partir de quelques filings SEC
    docs = load_raw_documents_from_sec(
        dataset_path="data/raw/sec_filings_1000.csv",
        limit=5,
    )

    parent_child = build_parent_child_chunks(docs, max_chars=800, overlap_chars=150)
    child_chunks = parent_child.child_chunks

    cfg = VectorIndexConfig(
        persist_dir=Path("data/vector_store/sec_demo_index")
    )

    idx = build_vector_index(
        documents=child_chunks,
        config=cfg,
    )

    print("Index construit et persisté dans", cfg.persist_dir)
