# TrustRAG — Retrieval-Augmented Generation Fiable pour l’Analyse Financière

## 🎯 Objectif du projet


**TrustRAG** est un système de **Retrieval-Augmented Generation (RAG)** conçu pour répondre à un défi clé :

> **Entre deux informations similaires, comment choisir la source la plus fiable ?**  
> (et éviter que le LLM réponde à partir d’un document moins autoritaire)

Contrairement au RAG classique, qui repose uniquement sur la similarité d’embeddings, TrustRAG ajoute une notion cruciale :

La fiabilité des sources (authority-aware retrieval)**

Ainsi, entre deux documents très proches :

 - Un formulaire SEC audité → priorité
    
 - Un article Web ou une brève de presse → utilisé uniquement en contexte secondaire

TrustRAG est donc un **RAG sensible à l’autorité, la fraîcheur, et la structure de l’information**.


#  1. Pourquoi TrustRAG ?
Les embeddings ne suffisent pas :

- une brochure commerciale peut être **plus similaire** qu’un rapport SEC,
    
- une news récente peut être **plus proche textuellement** qu’un bilan financier.

→ Le RAG classique choisit la mauvaise source.  
→ TrustRAG introduit **la qualité comme critère principal**.

Ce projet résout donc directement le défi donné.
---

#  2. Bases de données utilisées (2 niveaux d’autorité)

TrustRAG exploite **deux sources de données distinctes**, chacune avec un **score d'autorité** intégré dans le pipeline.


## Tier 1 — Base financière SEC (haute autorité)

Contenu :

- filings 10-K / 10-Q
- données financières officielles
- triples structurés (revenue, debt, assets…)
- score d’autorité élevé (1.0)

Rôle :

- fournir les faits comptables exacts
- base prioritaire pour les chiffres critiques
- alignée avec l’objectif : favoriser la source la plus fiable


## Tier 2 — Base Market News & Macro Trends (autorité moyenne)

Dataset fourni :

- **actualités économiques** (headline, date)  
- **indice boursier impacté** (S&P500, Shanghai Composite…) 
- **variation (%)**
- **sentiment**
- **secteur concerné**
- **impact_level (Low / Medium / High)**
- **entreprise associée**

   Rôle :

- contextualiser une variation
- apporter du macro (inflation, housing, FX…)
- **jamais remplacer un chiffre officiel**

---

## Pourquoi c’est important ?
Pour une question comme :
> “Pourquoi la dette d’Apple augmente ?”

- **Tier1 SEC** → valeur exacte de la dette 
- **Tier2 News** → contexte macro pouvant expliquer la tendance
TrustRAG combine les deux de façon contrôlée.



graph TD
    A[app/gui_gradio.py] --> B(core/ingestion)
    B --> B1(loaders.py)
    B --> B2(chunker.py)
    B --> B3(metadata.py)

    A --> C(core/index)
    C --> C1(index_manager.py)

    A --> D(core/knowledge_graph)
    D --> D1(kg_builder.py)
    D --> D2(kg_client.py)

    A --> E(core/retrieval)
    E --> E1(dual_retriever.py)
    E --> E2(reranker_trust.py)
    E --> E3(query_transformer.py)

    A --> F(core/generation)
    F --> F1(generator.py)
    F --> F2(grounding_guardrails.py)

    A --> G(pipelines/retrieval_pipeline.py)


