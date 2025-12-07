from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from datetime import datetime
from enum import Enum

import pandas as pd
import requests
from bs4 import BeautifulSoup
from llama_index.core import Document

import re
import language_tool_python
from langdetect import detect, LangDetectException
import fitz

#LT_FR = language_tool_python.LanguageTool("fr")
LT_FR = None

#LT_EN = language_tool_python.LanguageTool("en-US")
T_EN =  None

logger = logging.getLogger(__name__)


SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

USER_AGENT = "TrustRAG/0.1 (cikram.jemlaoui3@gmail.com)"

# Petit échantillon par défaut (Apple, Microsoft)
DEFAULT_SAMPLE_CIKS = ["0000320193", "0000789019"]  # Apple, Microsoft
# élargi pour récupérer plus de lignes
DEFAULT_SAMPLE_FORMS = ["10-K", "10-Q", "8-K"]


def _detect_lang(text: str) -> str:
    """Retourne 'fr' ou 'en' (fallback 'en' si doute)."""
    try:
        lang = detect(text)
    except LangDetectException:
        return "en"
    if lang.startswith("fr"):
        return "fr"
    return "en"


def grammar_quality_score(text: str) -> float:
    if not text:
        return 0.0

    words = re.findall(r"\w+", text)
    n_words = len(words) or 1

    lang = _detect_lang(text)
    tool = LT_FR if lang == "fr" else LT_EN

    matches = tool.check(text)
    ratio = len(matches) / n_words  # erreurs / mot

    # moins d’erreurs => meilleur score
    if ratio == 0:
        return 1.0
    elif ratio < 0.01:   # < 1 erreur / 100 mots
        return 0.8
    elif ratio < 0.03:   # < 3 erreurs / 100 mots
        return 0.6
    elif ratio < 0.07:   # < 7 erreurs / 100 mots
        return 0.4
    else:
        return 0.2        # beaucoup de fautes

# ----------------------------- Modélisation des catégories d'autorité (V4-A)-----------------------------------------

class AuthorityCategory(str, Enum):
    REGULATORY_AUDITED = "regulatory_audited"          # 1.0
    ACADEMIC_PEER_REVIEWED = "academic_peer_reviewed"  # 0.8
    OFFICIAL_OPERATIONAL = "official_operational"      # 0.5
    PUBLIC_GENERAL = "public_general"                  # 0.3


def infer_authority_score_base(source: str, url: Optional[str]) -> Tuple[float, str]:
   
    # Déduir le score V4-A  à partir du type de source 
    
    src = (source or "").lower().strip()
    url_l = (url or "").lower()

    # 1 Réglementaire  1.0
    if (
        "sec_edgar" in src
        or "sec.gov" in url_l
        or "insee" in src
        or "sirene" in src
        or "amf" in src
        or "amf-france" in url_l
        or  "gouv" in url_l
        or  url_l.endswith(".gouv.fr")

    ):
        return 1.0, AuthorityCategory.REGULATORY_AUDITED.value

    # 2️ Académique / Peer-review 0.8
    if (
        "hal" in src
        or "theses.fr" in url_l
        or "hal.science" in url_l
        or "pubmed" in url_l
        or "doi.org" in url_l
    ):
        return 0.8, AuthorityCategory.ACADEMIC_PEER_REVIEWED.value

    # 3️ Opérationnel 0.5
    if (
         "who.int" in url_l
        or "europa.eu" in url_l
        or "official" in src
    ):
        return 0.5, AuthorityCategory.OFFICIAL_OPERATIONAL.value

    # 4️ Dataset de news financières
    if "financial_news_events" in src or "financial_impact_ai_lab" in src:
        return 0.3, AuthorityCategory.PUBLIC_GENERAL.value

    # 5️ Autre
    return 0.0, AuthorityCategory.PUBLIC_GENERAL.value


# --------------------------------- Helpers EDGAR-------------------------------------

def _normalize_cik(cik: str | int) -> str:
    return f"{int(str(cik).strip()):010d}"


def _get_sec_json_for_cik(cik: str) -> dict:
    """Télécharge le JSON de soumissions EDGAR pour un CIK donné."""
    cik_norm = _normalize_cik(cik)
    url = SEC_SUBMISSIONS_URL.format(cik=cik_norm)
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _download_filing_html(cik: str, accession: str, primary_doc: str) -> Tuple[str, str]:
   
    cik_no_zero = str(int(cik))  # sans zéros en tête
    acc_no_dash = accession.replace("-", "")
    url = f"{SEC_ARCHIVES_BASE}/{cik_no_zero}/{acc_no_dash}/{primary_doc}"

    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    # on enlève scripts / styles
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = " ".join(soup.get_text(" ").split())
    return text, url


# -----------------------------------TÉLÉCHARGEMENT DE VRAIES DONNÉES EDGAR → CSV LOCAL-----------------------------------

def download_sec_filings_from_edgar(
    ciks: Sequence[str],
    form_types: Optional[List[str]],
    limit_per_cik: int,
    output_csv: str | Path,
    sec_request_pause_s: float = 0.3,
) -> Path:
    
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []

    # Normalisation des types de formulaires
    form_types_norm = None
    if form_types:
        form_types_norm = {ft.upper().strip() for ft in form_types}

    for cik in ciks:
        try:
            logger.info(f"[EDGAR] Récupération du JSON pour CIK={cik}")
            data = _get_sec_json_for_cik(cik)
        except Exception as e:
            logger.error("Erreur JSON EDGAR pour CIK=%s : %s", cik, e)
            continue

        company_name = data.get("name")
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        filing_dates = recent.get("filingDate", [])

        count = 0
        for form, acc, doc, fdate in zip(
            forms, accessions, primary_docs, filing_dates
        ):
            if form_types_norm and form.upper().strip() not in form_types_norm:
                continue
            if count >= limit_per_cik:
                break

            try:
                logger.info(
                    f"[EDGAR] Téléchargement CIK={cik} form={form} accession={acc}"
                )
                text, url = _download_filing_html(cik, acc, doc)
                time.sleep(sec_request_pause_s)
            except Exception as e:
                logger.error(
                    "Erreur téléchargement filing CIK=%s accession=%s : %s",
                    cik,
                    acc,
                    e,
                )
                continue

            if not text.strip():
                continue

            cik_norm = _normalize_cik(cik)
            source_system = "sec_edgar"
            authority_score_base, authority_category = infer_authority_score_base(
                source_system, url
            )
            ingestion_ts = datetime.utcnow().isoformat()
            trace_id = f"{source_system}::{cik_norm}::{fdate}::{form}"

            rows.append(
                {
                    "cik": cik_norm,
                    "company_name": company_name,
                    "form_type": form,
                    "filing_date": fdate,
                    "text": text,
                    "url": url,
                    "source_system": source_system,
                    "authority_score_base": authority_score_base,
                    "authority_category": authority_category,
                    "ingestion_timestamp": ingestion_ts,
                    "trace_id": trace_id,
                }
            )
            count += 1

        logger.info("CIK=%s → %d filings retenus", cik, count)

    if not rows:
        raise RuntimeError(
            "Aucun filing EDGAR récupéré. Vérifie User-Agent, CIKs ou réseau."
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    logger.info(" Dataset EDGAR sauvegardé : %s (lignes=%d)", output_csv, len(df))
    return output_csv


# -----------------------------------LECTURE DU CSV LOCAL SEC → Documents LlamaIndex-----------------------------------

def load_local_sec_filings(
    path: str | Path,
    form_types: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> List[Document]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset SEC introuvable : {path}")

    logger.info(f"Chargement du dataset SEC local depuis {path}")

    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Aucun fichier CSV trouvé dans le dossier {path}")
        frames = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(path)

    df.columns = [str(c).lower().strip() for c in df.columns]

    required_cols = ["cik", "form_type", "filing_date", "text"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Colonne requise manquante dans le dataset SEC : '{col}'. "
                f"Colonnes trouvées : {list(df.columns)}"
            )

    if form_types:
        form_types_norm = {ft.upper().strip() for ft in form_types}
        before = len(df)
        df = df[df["form_type"].str.upper().isin(form_types_norm)]
        logger.info(
            "Filtrage par form_types=%s : %d -> %d lignes",
            list(form_types_norm),
            before,
            len(df),
        )

    if limit is not None:
        df = df.head(limit)
        logger.info("Limitation à %d filings", len(df))

    documents: List[Document] = []

    for _, row in df.iterrows():
        text = row.get("text", "")
        if pd.isna(text) or not str(text).strip():
            continue

        url = row.get("url") or row.get("filing_url")
        source_system = row.get("source_system") or "sec_edgar"

        if "authority_score_base" in df.columns:
            authority_score_base = float(row.get("authority_score_base", 0.3))
            authority_category = str(
                row.get("authority_category") or AuthorityCategory.PUBLIC_GENERAL.value
            )
        else:
            authority_score_base, authority_category = infer_authority_score_base(
                source_system, url
            )

        trace_id = row.get("trace_id")
        if pd.isna(trace_id) or not trace_id:
            trace_id = f"{source_system}::{row.get('cik')}::{row.get('filing_date')}::{row.get('form_type')}"

        metadata = {
            "source": "sec_edgar",
            "source_system": source_system,
            "source_url": url,
            "trace_id": trace_id,
            "source_authority_score_base": authority_score_base,
            "source_authority_category": authority_category,
            "cik": str(row.get("cik")),
            "company_name": row.get("company_name") or row.get("company"),
            "form_type": row.get("form_type"),
            "filing_date": str(row.get("filing_date")),
        }

        documents.append(Document(text=str(text), metadata=metadata))

    logger.info("Documents SEC construits : %d", len(documents))
    return documents


# ---------------------------------DEUXIEME SOURCE : Financial News Events (CSV) → Documents LlamaIndex-------------------------------------

def load_local_financial_news_events(
    path: str | Path,
    limit: Optional[int] = None,
) -> List[Document]:
    """
    Dataset attendu :
      - 'index'
      - 'Date'
      - 'Headline'
      - 'Source'
      - 'Market_Event'
      - 'Market_Index'
      - 'Index_Change_Percent'
      - 'Trading_Volume'
      - 'Sentiment'
      - 'Sector'
      - 'Impact_Level'
      - 'Related_Company'
      - 'News_Url'
      - 'Encoded_Impact_Level'
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset news introuvable : {path}")

    logger.info(f"Chargement du dataset Financial News Events depuis {path}")
    df = pd.read_csv(path)

    cols = list(df.columns)
    required_cols = ["Date", "Headline"]
    for col in required_cols:
        if col not in cols:
            raise ValueError(
                f"Colonne requise manquante pour les news : '{col}'. "
                f"Colonnes trouvées : {cols}"
            )

    if limit is not None:
        df = df.head(limit)
        logger.info("Limitation à %d news", len(df))

    documents: List[Document] = []
    source_system = "financial_impact_ai_lab"

    for _, row in df.iterrows():
        text = row.get("Headline", "")
        if pd.isna(text) or not str(text).strip():
            continue

        event_date_raw = row.get("Date")
        event_date_str = None if pd.isna(event_date_raw) else str(event_date_raw)

        news_source_name = row.get("Source")
        market_event = row.get("Market_Event")
        market_index = row.get("Market_Index")
        index_change_percent = row.get("Index_Change_Percent")
        trading_volume = row.get("Trading_Volume")
        sentiment_val = row.get("Sentiment")
        sector_val = row.get("Sector")
        impact_level = row.get("Impact_Level")
        related_company = row.get("Related_Company")
        news_url = row.get("News_Url")

        authority_score_base, authority_category = infer_authority_score_base(
            source_system, str(news_url) if news_url is not None else None
        )

        trace_id = f"{source_system}::{event_date_str}::{related_company or 'NA'}"

        metadata = {
            "source": "financial_news_events",
            "source_system": source_system,
            "source_url": news_url,
            "trace_id": trace_id,
            "source_authority_score_base": authority_score_base,
            "source_authority_category": authority_category,
            "event_date": event_date_str,
            "headline_source": news_source_name,
            "market_event": market_event,
            "market_index": market_index,
            "index_change_percent": index_change_percent,
            "trading_volume": trading_volume,
            "sentiment": sentiment_val,
            "sector": sector_val,
            "impact_level": impact_level,
            "related_company": related_company,
            "dataset_name": "financial_impact_ai_lab",
        }

        documents.append(Document(text=str(text), metadata=metadata))

    logger.info("Documents Financial News Events construits : %d", len(documents))
    return documents


def load_raw_documents_from_financial_news(
    dataset_path: str | Path = "data/raw/financial_impact_ai_lab.csv",
    limit: Optional[int] = None,
) -> List[Document]:
    """
    Point d'entrée pipeline pour les news financières.
    """
    return load_local_financial_news_events(
        path=dataset_path,
        limit=limit,
    )


# -----------------------------Point d'entrée SEC utilisé par le pipeline d’ingestion-----------------------------------------

def load_raw_documents_from_sec(
    dataset_path: str | Path = "data/raw/sec_filings_sample.csv",
    form_types: Optional[List[str]] = None,
    limit: Optional[int] = None,
    limit_per_cik: int = 5,
    ciks: Optional[Sequence[str]] = None,
) -> List[Document]:
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        logger.info(
            "Dataset %s introuvable → téléchargement d’un échantillon EDGAR...",
            dataset_path,
        )
        download_sec_filings_from_edgar(
            ciks=ciks or DEFAULT_SAMPLE_CIKS,
            form_types=form_types or DEFAULT_SAMPLE_FORMS,
            limit_per_cik=limit_per_cik,
            output_csv=dataset_path,
        )

    return load_local_sec_filings(
        path=dataset_path,
        form_types=form_types,
        limit=limit,
    )


# ------------------------------------- PDF SCORE---------------------------------

def load_web_sources(
    urls: Sequence[str],
) -> List[Document]:
    """
    Charge des documents à partir de liens web.
    Pour l'instant :
      - on se concentre sur les liens PDF
      - on applique un filtre grammaire uniquement pour la catégorie PUBLIC_GENERAL
    """
    documents: List[Document] = []

    for url in urls:
        url_str = str(url).strip()
        url_l = url_str.lower()

        
        if not url_l.endswith(".pdf"):
            
            continue

        #  Télécharger le PDF
        try:
            resp = requests.get(url_str, headers={"User-Agent": USER_AGENT}, timeout=60)
            resp.raise_for_status()
        except Exception as e:
            logger.error("Erreur téléchargement PDF web %s : %s", url_str, e)
            continue

        #  Extraire le texte du PDF
        try:
            with fitz.open(stream=resp.content, filetype="pdf") as pdf_doc:
                pages_text = [page.get_text() for page in pdf_doc]
            full_text = "\n\n".join(pages_text).strip()
        except Exception as e:
            logger.error("Erreur extraction texte PDF web %s : %s", url_str, e)
            continue

        if not full_text:
            continue

        #  Déterminer la catégorie d'autorité à partir de la source + URL
        source_system = "web_pdf"
        authority_score_base, authority_category = infer_authority_score_base(
            source_system,
            url_str,
        )

        #  Si catégorie = PUBLIC_GENERAL -> filtrage grammaire FR/EN
        if authority_category == AuthorityCategory.PUBLIC_GENERAL.value:
            gram_score = grammar_quality_score(full_text)

            # règle : si grammaire/conjugaison/cohérence < 0.5 -> on n'utilise pas ce doc
            if gram_score < 0.5:
                logger.info(
                    "PDF web ignoré (cat=PUBLIC_GENERAL, gram=%.2f, url=%s)",
                    gram_score,
                    url_str,
                )
                continue

            # si gram >= 0.5 : on garde score 0.3 (déjà fixé par la catégorie)
            authority_score_base = 0.3

        #  Construction des métadonnées communes
        trace_id = f"{source_system}::{url_str}"

        metadata = {
            "source": "web_pdf",
            "source_system": source_system,
            "source_url": url_str,
            "trace_id": trace_id,
            "source_authority_score_base": authority_score_base,
            "source_authority_category": authority_category,
        }

        documents.append(Document(text=full_text, metadata=metadata))

    logger.info("Documents web (PDF) construits : %d", len(documents))
    return documents

