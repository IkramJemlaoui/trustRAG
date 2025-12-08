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



trustRAG/
│
├── app/
│   └── gui_gradio.py                # Interface utilisateur (Gradio)
│
├── core/
│   ├── ingestion/
│   │   ├── loaders.py               # Chargement & parsing des documents
│   │   ├── chunker.py               # Découpage intelligent en chunks
│   │   └── metadata.py              # Calcul des scores V4-A
│   │
│   ├── index/
│   │   └── index_manager.py         # Vector store (embeddings)
│   │
│   ├── knowledge_graph/
│   │   ├── kg_builder.py            # Construction du Knowledge Graph
│   │   └── kg_client.py             # Recherche dans le KG structuré
│   │
│   ├── retrieval/
│   │   ├── dual_retriever.py        # Fusion vectorielle + KG
│   │   ├── reranker_trust.py        # Reranking basé sur la fiabilité
│   │   └── query_transformer.py     # Génération de variantes de requêtes (désactivé)
│   │
│   └── generation/
│       ├── generator.py             # Appel LLM (Ollama)
│       └── grounding_guardrails.py  # Guardrails anti-hallucination
│
└── pipelines/
    └── retrieval_pipeline.py        # Pipeline retrieval + génération
