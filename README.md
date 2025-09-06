# Sparse FTT+ : Feature Tokenizer Transformer avec Attention Multi-Têtes Interprétable pour Données Tabulaires

---

## 1. Introduction

Les méthodes de deep learning appliquées aux données tabulaires montrent des performances prometteuses, mais leur adoption opérationnelle reste limitée par un déficit d'interprétabilité et par l'hétérogénéité des protocoles expérimentaux. La représentation transparente des décisions est essentielle pour garantir la confiance des modèles dans des contextes critiques. Dans cette optique, notre travail vise à enrichir l'arsenal des architectures interprétables en proposant une nouvelle variante du Feature Tokenizer Transformer.

Notre approche repose sur la conception du **Sparse FT-Transformer+ (Sparse FTT+)**, combinant la structure du FT-Transformer avec des mécanismes d'attention interprétables inspirés du Temporal Fusion Transformer. Cette nouvelle architecture offre une adaptabilité accrue pour concilier performance prédictive et transparence décisionnelle grâce à l'intégration d'une attention multi-têtes interprétable utilisant l'opérateur sparsemax.

Sparse FTT+ est particulièrement adapté aux applications nécessitant une explication claire des prédictions, comme dans la banque, l'assurance, ou la santé, où la compréhension des décisions algorithmiques est cruciale pour l'acceptation et la conformité réglementaire.

---

## 2. Innovation Principale : Attention Multi-Têtes Interprétable

### 2.1 Problématique de l'interprétabilité dans les Transformers

Les architectures Transformer traditionnelles souffrent d'un manque de transparence dans leurs mécanismes d'attention, particulièrement problématique pour les données tabulaires où l'identification des features influentes est cruciale. Les têtes d'attention multiples avec des matrices de valeurs (V) distinctes créent des représentations complexes difficiles à interpréter.

### 2.2 Solution proposée : Attention Interprétable

**Sparse FTT+** introduit une architecture d'attention révolutionnaire qui résout ces limitations :

#### 2.2.1 Valeur Partagée (V) entre Têtes
- **Innovation clé** : Une seule matrice de valeur (V) partagée entre toutes les têtes d'attention
- **Avantage** : Élimine les distortions dues à des transformations V différentes
- **Résultat** : Les scores d'attention deviennent directement comparables et interprétables

#### 2.2.2 Agrégation Significative des Scores
- **Moyenne pondérée** : Les scores d'attention sont moyennés across les têtes pour refléter l'importance réelle de chaque feature
- **Transparence** : Chaque score représente fidèlement la contribution d'une feature à la décision finale
- **Comparabilité** : Les scores entre différentes features sont directement comparables

#### 2.2.3 Attention Parcimonieuse via Sparsemax
- **Outil au service de l'interprétabilité** : Sparsemax remplace softmax pour produire des distributions d'attention creuses
- **Bénéfice** : Concentration automatique sur les features les plus pertinentes
- **Clarté** : Élimination du bruit attentionnel sur les features non contributives

---

## 3. Architecture Technique du Sparse FTT+

### 3.1 Schéma global du forward pass

<div align="center">
  <img src="images/FT_Transformer architecture.png" alt="Architecture globale du Sparse FTT+ appliqué aux données tabulaires" width="500"/>
  <br>
  <b>Architecture globale du Sparse FTT+ appliqué aux données tabulaires</b>
  <br>
  Source: Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2023). <i>Revisiting Deep Learning Models for Tabular Data</i>. arXiv:2106.11959.
</div>

### 3.2 Composants Techniques

#### 3.2.1 Tokenisation des Features

Le [`FeatureTokenizer`](rtdl_revisiting_models/lib/deep.py) encode les variables numériques et catégoriques en vecteurs denses de dimension `d_token` :
- **Variables numériques** : Transformées via une projection linéaire ou transformation personnalisée
- **Variables catégoriques** : Encodées en embeddings appris selon les cardinalités

<div align="center">
  <img src="images/Illustration%20d'un%20Feature%20Tokenizer.png" alt="Illustration du processus de tokenisation des variables brutes en vecteurs denses" width="500"/>
  <br>
  <b>Tokenisation des features : transformation en séquence uniforme pour le Transformer</b>
  <br>
  Source: Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2023). <i>Revisiting Deep Learning Models for Tabular Data</i>. arXiv:2106.11959.
</div>

#### 3.2.2 Token CLS Agrégateur

Un token spécial [`[CLS]`](rtdl_revisiting_models/bin/sparse_ftt_plus.py) appris agrège les informations des features et sert de base pour la prédiction finale.

#### 3.2.3 Blocs Transformer Interprétables

Chaque bloc applique l'attention multi-têtes interprétable, suivi d'un FFN et de la normalisation résiduelle.

##### Mécanisme d'Attention Interprétable

<div align="center">
  <img src="images/Scaled Dot-Product Attention.png" alt="Scaled Dot-Product Attention avec sparsemax pour Sparse FTT+" width="300"/>
  <br>
  <b>Attention Parcimonieuse : Sparsemax au service de l'interprétabilité</b>
</div>

---

<div align="center">
  <img src="images/Interpretable Multi-Head Self-Attention.png" alt="Illustration de l'Interpretable Multi-Head Self-Attention avec V partagé" width="500"/>
  <br>
  <b>Innovation Principale : Attention Multi-Têtes avec V Partagé pour l'Interprétabilité</b>
</div>

---

##### Feed-Forward Network et Normalisation

<div align="center">
  <img src="images/One Transformer layer.png" alt="Vue d'ensemble d'un bloc Transformer adapté aux données tabulaires (Sparse FTT+)" width="300"/>
  <br>
  <b>Bloc Transformer Interprétable pour Données Tabulaires</b>
</div>

---

### 3.3 Extraction de l'Importance des Features

#### 3.3.1 Scores d'Attention Directs
- **Source** : Matrice d'attention CLS→features, directement interprétable grâce au V partagé
- **Normalisation** : Scores normalisés et souvent creuses via sparsemax
- **Fiabilité** : Reflet direct des contributions sans distortion

#### 3.3.2 Avantages de l'Architecture Interprétable
- **Transparence** : Identification claire des features déterminantes
- **Parcimonie** : Élimination automatique des features non pertinentes
- **Comparabilité** : Scores directement comparables entre features et échantillons

---

## 4. Positionnement et Contributions

### 4.1 Innovation par rapport au FT-Transformer
- **Attention interprétable** : V partagé et agrégation meaningful des scores
- **Transparence accrue** : Mécanisme d'attention directement exploitable
- **Inspiration TFT** : Adaptation des concepts d'interprétabilité du Temporal Fusion Transformer

### 4.2 Objectifs de Recherche
- **Lever le "black box effect"** : Rendre les décisions des modèles tabulaires explicables
- **Performance et transparence** : Maintenir les performances tout en garantissant l'interprétabilité
- **Applications critiques** : Répondre aux besoins d'explication en banque, assurance, santé

### 4.3 Architecture Réutilisable
- **Code modulaire** : Implémentation flexible pour diverses applications tabulaires
- **Intégration facile** : Compatible avec les pipelines de données existants
- **Extensibilité** : Base pour futures innovations en interprétabilité

---

## 5. Références

- Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. NeurIPS.
- Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). *Revisiting Deep Learning Models for Tabular Data*. arXiv:2106.11959.
- Lim, B., Arik, S. Ö., Loeff, N., & Pfister, T. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. NeurIPS.
- Martins, A. F. T., & Astudillo, R. F. (2016). *From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification*. ICML.
- Gorishniy, Y., Rubachev, I., & Babenko, A. (2021). *On Embeddings for Numerical Features in Tabular Deep Learning*. NeurIPS.
- Devlin, J., et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.

---

## Auteur

**Léonel VODOUNOU**  
Sparse FTT+ – Architecture Interprétable pour l'Apprentissage sur Données Tabulaires  
2025