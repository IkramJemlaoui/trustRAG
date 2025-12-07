from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
from core.ingestion.loaders import infer_authority_score_base

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SEC_CSV = PROJECT_ROOT / "data" / "raw" / "sec_filings_1000.csv"
DEFAULT_KG_JSON = PROJECT_ROOT / "data" / "graph_store" / "kg_facts.json"


# -------------------------------------------------------------------
# 1. Dictionnaire GAAP (normalisation de labels)
# -------------------------------------------------------------------
GAAP_MAP = {
    "net sales": "Revenue",
    "sales": "Revenue",
    "revenue": "Revenue",
    "total revenue": "Revenue",
    "operating income": "OperatingIncome",
    "income": "Income",
    "assets": "Assets",
    "liabilities": "Liabilities",
    "total liabilities": "Liabilities",
    "long term debt": "LongTermDebt",
    "long-term debt": "LongTermDebt",
    "cash": "Cash",
}


def normalize_label(label: str) -> Optional[str]:
    """Essaie de mapper une ligne texte (label) vers un concept GAAP standard."""
    label = label.strip().lower()
    for k, std in GAAP_MAP.items():
        if k in label:
            return std
    return None


# ---------------------------Dataclass TRIPLE----------------------------------------

@dataclass
class KGTriple:
    id: str
    subject: str
    relation: str
    object: Any
    authority_score: float
    metadata: Dict[str, Any]


# ---------------------Extraction de valeurs numériques + parsing structuré-------------------------------------

NUMERIC_PATTERN = re.compile(r"([\$€]?\s?\d[\d,\.]+)")


def extract_numeric_value(text: str) -> Optional[str]:
    """Retourne la première occurrence d'un nombre style '$ 1,234.56' ou '1234' """
    m = NUMERIC_PATTERN.search(text)
    if not m:
        return None
    return m.group(1)


def parse_numeric_to_float(num_str: str) -> Optional[float]:
    """Convertit une chaîne numérique en float (enlevant $, €, virgules)."""
    try:
        cleaned = (
            num_str.replace("$", "")
            .replace("€", "")
            .replace(",", "")
            .strip()
        )
        # gestion simple du point décimal
        return float(cleaned)
    except Exception:
        return None


def infer_currency(num_str: str) -> Optional[str]:
    """Devine la devise à partir du symbole (très basique)."""
    if "$" in num_str:
        return "USD"
    if "€" in num_str:
        return "EUR"
    # si rien, on ne sait pas (tu peux mettre 'USD' si tu veux forcer)
    return None


# ---------------------------------Détection d'indices macro-économiques dans le texte-------------------------

MACRO_TERMS = {
    "inflation": "Inflation",
    "interest rate": "InterestRates",
    "interest rates": "InterestRates",
    "foreign exchange": "Forex",
    "fx": "Forex",
    "macroeconomic": "MacroGeneral",
    "macroeconomic conditions": "MacroGeneral",
    "recession": "Recession",
    "economic downturn": "Recession",
    "global demand": "GlobalDemand",
    "demand slowdown": "DemandSlowdown",
    "supply chain": "SupplyChain",
    "logistics": "SupplyChain",
    "trade war": "TradePolicy",
    "tariff": "TradePolicy",
    "tariffs": "TradePolicy",
}


def detect_macro_signals(line: str) -> List[str]:
    """Retourne une liste de tags macro trouvés dans la ligne."""
    tags: List[str] = []
    ln = line.lower()
    for key, tag in MACRO_TERMS.items():
        if key in ln:
            tags.append(tag)
    return list(set(tags))  # on enlève les doublons


# -------------------------------Utilitaires sur la période (année, trimestre, type)-------------------------

def infer_period_from_filing(filing_date: str, form_type: str) -> Dict[str, Any]:
    
    try:
        dt = datetime.strptime(str(filing_date), "%Y-%m-%d")
        year = dt.year
        month = dt.month
    except Exception:
        # si le format n'est pas propre, on renvoie quelque chose de minimal
        return {
            "period_year": None,
            "period_quarter": None,
            "period_type": "Unknown",
        }

    form = (form_type or "").upper()
    if form.startswith("10-K"):
        period_type = "Annual"
        period_quarter = None
    elif form.startswith("10-Q"):
        period_type = "Quarterly"
        # mapping simplifié mois → trimestre
        if month in (1, 2, 3):
            period_quarter = "Q1"
        elif month in (4, 5, 6):
            period_quarter = "Q2"
        elif month in (7, 8, 9):
            period_quarter = "Q3"
        else:
            period_quarter = "Q4"
    else:
        period_type = "Other"
        period_quarter = None

    return {
        "period_year": year,
        "period_quarter": period_quarter,
        "period_type": period_type,
    }


# -------------------------------Construction du KG structuré depuis un CSV SEC---------------------------------

def build_structured_kg_from_sec(
    sec_csv_path: str | Path = DEFAULT_SEC_CSV,
    output_path: str | Path = DEFAULT_KG_JSON,
    max_triples_per_doc: int = 50,
    max_macro_triples_per_doc: int = 20,
) -> Path:
    
    sec_csv_path = Path(sec_csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not sec_csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable : {sec_csv_path}")

    logger.info("Chargement SEC CSV : %s", sec_csv_path)
    df = pd.read_csv(sec_csv_path)
    df.columns = [c.lower().strip() for c in df.columns]

    required = ["text", "company_name", "cik", "filing_date", "form_type"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column missing: {col}")

    triples: List[KGTriple] = []
    triple_id = 0

    for _, row in df.iterrows():
        text = str(row.get("text") or "")

        company = row.get("company_name")
        cik = row.get("cik")
        filing_date = row.get("filing_date")
        form = row.get("form_type")
        url = row.get("url") or row.get("filing_url")

        authority, cat = infer_authority_score_base("sec_edgar", url)
        period_info = infer_period_from_filing(filing_date, form)

        # on découpe le texte par lignes (utiles pour repérer les tableaux / phrases)
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        doc_triples: List[KGTriple] = []
        macro_triples: List[KGTriple] = []

        # extraction des valeurs numériques structurées
        for ln in lines:
            numeric_raw = extract_numeric_value(ln)
            if not numeric_raw:
                continue

            gaap_concept = normalize_label(ln)
            if not gaap_concept:
                continue

            numeric_value = parse_numeric_to_float(numeric_raw)
            if numeric_value is None:
                continue

            currency = infer_currency(numeric_raw)

            triple_id += 1
            # on garde la date dans la relation pour différencier les années
            rel = f"{gaap_concept}_{filing_date}"

            triple = KGTriple(
                id=f"triple_{triple_id}",
                subject=company,
                relation=rel,
                object=numeric_raw,  # on garde la version texte en "object"
                authority_score=authority,
                metadata={
                    "cik": cik,
                    "filing_date": filing_date,
                    "form_type": form,
                    "source_url": url,
                    "raw_text": ln,
                    "gaap_concept": gaap_concept,
                    "numeric_raw": numeric_raw,
                    "numeric_value": numeric_value,
                    "currency": currency,
                    "authority_category": cat,
                    **period_info,
                },
            )
            doc_triples.append(triple)

            if len(doc_triples) >= max_triples_per_doc:
                break

        # extraction de signaux macro-économiques
        for ln in lines:
            tags = detect_macro_signals(ln)
            if not tags:
                continue

            for tag in tags:
                triple_id += 1
                rel = f"MacroRisk_{tag}_{filing_date}"

                triple = KGTriple(
                    id=f"triple_{triple_id}",
                    subject=company,
                    relation=rel,
                    object=True,  # on met juste un booléen, le détail est dans raw_text
                    authority_score=authority,
                    metadata={
                        "cik": cik,
                        "filing_date": filing_date,
                        "form_type": form,
                        "source_url": url,
                        "raw_text": ln,
                        "macro_tag": tag,
                        "authority_category": cat,
                        **period_info,
                    },
                )
                macro_triples.append(triple)

                if len(macro_triples) >= max_macro_triples_per_doc:
                    break
            if len(macro_triples) >= max_macro_triples_per_doc:
                break

        triples.extend(doc_triples)
        triples.extend(macro_triples)

    # sauvegarde en JSON
    with output_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in triples], f, indent=2, ensure_ascii=False)

    logger.info(
        "KG structuré construit : %d triples (financiers + macro) → %s",
        len(triples),
        output_path,
    )
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_structured_kg_from_sec()
