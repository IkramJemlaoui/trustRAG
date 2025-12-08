# TrustRAG — Retrieval-Augmented Generation Fiable pour l’Analyse Financière

## 🎯 Objectif du projet


**TrustRAG** est un système de **Retrieval-Augmented Generation (RAG)** conçu pour répondre à un défi clé :

> **Entre deux informations similaires, comment choisir la source la plus fiable ?**  
> (et éviter que le LLM réponde à partir d’un document moins autoritaire)

Contrairement au RAG classique, qui repose uniquement sur la similarité d’embeddings, TrustRAG ajoute une notion cruciale :
























Les RAG classiques sont aveugles à la fiabilité. 
TrustRAG place la qualité, l’autorité et la fraîcheur des sources au centre du pipeline, afin de garantir des réponses justes, vérifiées et transparentes, même lorsque plusieurs documents semblent pertinents.

Problématique visée (challenge du projet)
“La similarité seule n’est pas suffisante. Entre deux informations similaires, le système doit toujours préférer la source la plus fiable.”

TrustRAG corrige ce biais en introduisant un pipeline qui :

 -   sélectionne d’abord les documents les plus fiables (via un score d’autorité),

-   combine retrieval vectoriel + faits structurés (KG),

-   génère une réponse vérifiée, traçable,

-    refuse de répondre si la confiance est insuffisante.

Exemple :
Entre deux informations similaires sur la dette d’Apple :

✔ SEC Form 10-K → haute autorité

✖ Blog  → faible autorité

TrustRAG doit éviter le blog même si l’embedding est plus proche.


**Fonctionnalités principales**
**1) Ingestion documentaire intelligente**

Ingestion documentaire intelligente

**Architecture du projet**
Base principale : SEC Vector Store (Tier 1 – haute autorité)

Contenu :

- filings 10-K / 10-Q

- données financières officielles

- triples structurés (revenue, debt, assets…)

- score d’autorité élevé (1.0)

Rôle :

- fournir les faits comptables exacts

- base prioritaire pour les chiffres critiques

- alignée avec l’objectif : favoriser la source la plus fiable


Base secondaire : Market News & Macro Dataset (Tier 2 – autorité moyenne)




trustRAG/
│
├── app/
│   └── gui_gradio.py        # Interface utilisateur
│
├── core/
│   ├── ingestion/
│   │   ├── loaders.py       # Chargement & parsing
│   │   ├── chunker.py       # Découpage intelligent
│   │   └── metadata.py      # Scores V4-A
│   │
│   ├── index/
│   │   └── index_manager.py # Vector store
│   │
│   ├── knowledge_graph/
│   │   ├── kg_builder.py    # Extraction des triples
│   │   └── kg_client.py     # Recherche dans le KG
│   │
│   ├── retrieval/
│   │   ├── dual_retriever.py# Fusion vector+KG
│   │   ├── reranker_trust.py# Reranking basé fiabilité
│   │   └── query_transformer.py (désactivé)
│   │
│   └── generation/
│       ├── generator.py     # Appel LLM
│       └── grounding_guardrails.py
│
└── pipelines/
    └── retrieval_pipeline.py# Pipeline complet


trustRAG/
│
├── app/
│   └── gui_gradio.py        # Interface utilisateur
│
├── core/
│   ├── ingestion/
│   │   ├── loaders.py       # Chargement & parsing
│   │   ├── chunker.py       # Découpage intelligent
│   │   └── metadata.py      # Scores V4-A
│   │
│   ├── index/
│   │   └── index_manager.py # Vector store
│   │
│   ├── knowledge_graph/
│   │   ├── kg_builder.py    # Extraction des triples
│   │   └── kg_client.py     # Recherche dans le KG
│   │
│   ├── retrieval/
│   │   ├── dual_retriever.py# Fusion vector+KG
│   │   ├── reranker_trust.py# Reranking basé fiabilité
│   │   └── query_transformer.py (désactivé)
│   │
│   └── generation/
│       ├── generator.py     # Appel LLM
│       └── grounding_guardrails.py
│
└── pipelines/
    └── retrieval_pipeline.py# Pipeline complet

