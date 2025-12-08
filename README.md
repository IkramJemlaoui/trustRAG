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


#  3. Architecture du pipeline

<img width="232" height="331" alt="image" src="https://github.com/user-attachments/assets/84a88c29-fdf6-46ff-887a-7b5cfa1656f1" />


#  4. Fonctionnalités principales

##  1) Ingestion intelligente

Extraction depuis :

- filings SEC (Tier 1) 
- news dataset (Tier 2)
- fichiers PDF / HTML (selon loaders)
    

Chaque chunk reçoit :

- source tier  
- date  
- présence de valeurs numériques 
- score d’autorité (basé sur la source EDGAR vs presse)
- métadonnées normalisées
    

---

## ✔ 2) Knowledge Graph structuré (KG)

Le fichier `kg_builder.py` :

- détecte les valeurs financières dans les filings SEC
- normalise les concepts GAAP (Revenue, Assets, Debt…)
- génère des triples structurés
- ajoute des triples macro (inflation, FX, supply chain…)
    

Le KG sert de base factuelle vérifiée pour la génération.

---

##  3) Dual Retrieval (Vector + KG)

Le fichier `dual_retriever.py` combine :

- **retrieval vectoriel**  
    → `bge-small-en-v1.5`
    
- **retrieval KG**  
    → matching lexical sur les triples
    

Résultat :
- passages de texte pertinents  
- chiffres exacts du KG  
- news pertinentes (Tier2)
    

---

##  4) Trust Re-ranking (V4-A)

Formule utilisée :

`score_final = 0.6 × similarité + 0.4 × autorité_source`

Motivation :
- 0.6 → assure que le passage répond vraiment à la question    
- 0.4 → impose la fiabilité en cas d’ambiguïté
    
Effet :
> Un passage SEC moins similaire > un blog plus proche vectoriellement.  
> (Concrètement observé dans les tests)

---

##   5) LLM Guardrails

Le module `grounding_guardrails.py` vérifie :
- autorité maximale du contexte 
- recouvrement lexical réponse ↔ sources 
- existence réelle des faits dans les documents
    

Si doute → refus :
> _"Je ne peux pas répondre de façon fiable avec les sources disponibles."_

---

##  6) LLM local via Ollama

Modèles utilisés :

- `qwen2.5:0.5b-instruct` 
- `phi3:mini`
    

Avantages :

- offline
- faible mémoire  
- adapté aux réponses factuelles



---

#  5. Structure du projet (pour GitHub)
    <img width="276" height="359" alt="image" src="https://github.com/user-attachments/assets/3ede1170-8670-47ab-adbd-54c4ab32109b" />


---

#  6. Démonstration (cas réel)

**Q : “Quel est le montant de la dette à long terme d’Apple en 2025 ?”**

Pipeline :
- KG → trouve le triple exact 
- Vector store → trouve paragraphes du 10-K  
- News dataset → ignoré (autorité faible)   
- Reranker → garde uniquement SEC  
- LLM → génère une réponse factuelle 
- Guardrails → validé
       "" Réponse exacte, vérifiée, non halluciné.""

---

#  7. Présentation KIP

## **K — Knowledge**

- Filings SEC
- Market News dataset
- KG structuré
- Scores d’autorité
    

## **I — Inputs**

- Question utilisateur
- Passages vectoriels
- Faits KG
- Métadonnées d’autorité
    

## **P — Processing**

- Dual retrieval
- Trust reranker
- LLM (Ollama)
- Guardrails anti-hallucination
    

## **Outputs**

- Réponse argumentée
- Faits sources affichés
- Score de confiance
- Possibilité de refus
    

## **Effects**

- Moins d’hallucinations
- Priorité aux sources expertes
- Transparence totale
    
---

#  8. Public cible idéal

🎯 **Analystes financiers**  
(VC, Private Equity, Hedge Funds, Corporate Finance)

Besoins couverts :
- extraction automatique de chiffres fiables
- comparaison multi-années
- contextualisation macro
- justification obligatoire des données
- zéro hallucination sur les valeurs financières
    

---

# 9. Limitations actuelles 

- LLM locaux → possible **timeout**
- KG uniquement basé sur SEC
- pas encore d’extraction tabulaire avancée (PDF)
- pas de NER financier spécialisé
- pas encore de multi-sources (Bloomberg / Yahoo Finance)
    

---

#  10. Améliorations prévues

- extraction de tableaux (Camelot / Tabula)
- ajout de multiples APIs financières
- embeddings spécialisés 
- NER financier (FinBERT / SpaCy)
- KG multi-années, multi-compagnies
- scoring d’autorité plus granulaire
    
---

# 11. Installation

`pip install -r requirements.txt pip install ollama`

**Télécharger les modèles Ollama :**

`ollama pull qwen2.5:0.5b-instruct ollama pull phi3:mini`

---

#  12. Lancement

## Construire le KG :
`python -m core.knowledge_graph.kg_builder`
## Ouvrir l’interface :
`python -m app.gui_gradio`



