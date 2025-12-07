**TrustRAG — Retrieval-Augmented Generation basé sur la Fiabilité des Sources**
TrustRAG est un Retrieval-Augmented Generator conçu pour résoudre un problème précis :
Dans un corpus contenant plusieurs informations similaires, comment s’assurer que la réponse provient de la source la plus fiable 
— et non de la source la plus “similaire” selon les embeddings ?

Les RAG classiques sont aveugles à la fiabilité. 
TrustRAG place la qualité, l’autorité et la fraîcheur des sources au centre du pipeline, afin de garantir des réponses justes, vérifiées et transparentes, 
même lorsque plusieurs documents semblent pertinents.

Problématique visée (challenge du projet)
“La similarité seule n’est pas suffisante. Entre deux informations similaires, le système doit toujours préférer la source la plus fiable.”

Objectif du projet

Construire un RAG qui privilégie la qualité, pas seulement la similarité.

Le système doit :

choisir les documents les plus fiables (notés via des métadonnées)

combiner récupération vectorielle + faits structurés

vérifier que la réponse est ancrée dans des sources crédibles

refuser de répondre si la fiabilité est insuffisante


Exemple :
Entre deux informations similaires sur la dette d’Apple :

✔ SEC Form 10-K → haute autorité

✖ Blog  → faible autorité

TrustRAG doit éviter le blog même si l’embedding est plus proche.


Fonctionnalités principales :
 **1) Ingestion documentaire intelligente**
Extraction depuis :

- Filings SEC (10-K, 10-Q)
    
- PDF financiers
    
- Pages HTML
    

Chaque chunk reçoit un score basé sur :

- source (SEC, corporate, web…)
    
- date
    
- qualité de structure
    
- présence de valeurs numériques
    
- contraintes métier


Ingestion documentaire intelligente
