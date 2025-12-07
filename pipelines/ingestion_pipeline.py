# core/ingestion/ingestion_pipeline.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from llama_index.core import Document

from core.ingestion.loaders import load_raw_documents_from_sec
from core.ingestion.chunker_advanced import build_parent_child_chunks
from core.index.index_manager import (
    build_vector_index,
    update_vector_index_with_documents,
    VectorIndexConfig,
)

logger = logging.getLogger(__name__)


# Config du pipeline d’ingestion

@dataclass
class IngestionPipelineConfig:

    sec_csv_path: Path = Path("data/raw/sec_filings_1000.csv")
    persist_dir: Path = Path("data/vector_store/sec_demo_index")

    max_chars: int = 800
    overlap_chars: int = 150

    limit_docs: Optional[int] = None
    mode_append: bool = False


# Pipeline d’ingestion SEC
class IngestionPipeline:
    def __init__(self, config: Optional[IngestionPipelineConfig] = None) -> None:
        self.config = config or IngestionPipelineConfig()

        logger.info(
            "IngestionPipeline initialisé (csv=%s, persist_dir=%s, mode_append=%s)",
            self.config.sec_csv_path,
            self.config.persist_dir,
            self.config.mode_append,
        )


    def _load_sec_documents(self) -> list[Document]:
        logger.info(
            "Chargement des filings SEC depuis %s (limit=%s)",
            self.config.sec_csv_path,
            self.config.limit_docs,
        )

        docs = load_raw_documents_from_sec(
            dataset_path=str(self.config.sec_csv_path),
            limit=self.config.limit_docs,
        )

        logger.info("Documents SEC chargés : %d", len(docs))
        return docs


    def _chunk_documents(self, docs: list[Document]):
        logger.info(
            "Construction des chunks (max_chars=%d, overlap_chars=%d)",
            self.config.max_chars,
            self.config.overlap_chars,
        )

        parent_child = build_parent_child_chunks(
            docs,
            max_chars=self.config.max_chars,
            overlap_chars=self.config.overlap_chars,
        )

        child_chunks = parent_child.child_chunks
        logger.info("Chunks enfants construits : %d", len(child_chunks))

        return child_chunks


    def _build_or_update_index(self, child_chunks: list[Document]) -> None:
        cfg = VectorIndexConfig(persist_dir=self.config.persist_dir)

        if self.config.mode_append and self.config.persist_dir.exists():
            logger.info(
                "Mode append ON → mise à jour de l’index existant dans %s",
                self.config.persist_dir,
            )
            update_vector_index_with_documents(
                documents=child_chunks,
                config=cfg,
            )
        else:
            logger.info(
                "Construction d’un NOUVEL index vectoriel dans %s",
                self.config.persist_dir,
            )
            build_vector_index(
                documents=child_chunks,
                config=cfg,
            )

        logger.info("Index vectoriel prêt dans %s", self.config.persist_dir)


    def run(self) -> None:
        """
        SEC CSV  -->  Documents  -->  Chunks enfants  -->  Index vectoriel
        """
        logger.info("=== DÉBUT PIPELINE D’INGESTION SEC ===")

        docs = self._load_sec_documents()
        if not docs:
            logger.warning("Aucun document SEC chargé, arrêt du pipeline.")
            return

        child_chunks = self._chunk_documents(docs)
        if not child_chunks:
            logger.warning("Aucun chunk généré, arrêt du pipeline.")
            return

        self._build_or_update_index(child_chunks)

        logger.info("=== FIN PIPELINE D’INGESTION SEC ===")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pipeline = IngestionPipeline(
        IngestionPipelineConfig(
            sec_csv_path=Path("data/raw/sec_filings_1000.csv"),
            persist_dir=Path("data/vector_store/sec_demo_index"),
            max_chars=800,
            overlap_chars=150,
            limit_docs=20,      
            mode_append=False,
        )
    )

    pipeline.run()
