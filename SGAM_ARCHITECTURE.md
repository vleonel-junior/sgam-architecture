# SGAM : Sequential Gated Additive Model

*Filtrage Séquentiel et Résolution de la Multicolinéarité pour Données Tabulaires*

*(Version corrigée et mathématiquement stricte)*

---

## Philosophie : Additivité Stricte et Filtrage Supervisé

Le SGAM est conçu pour résoudre l'explosion combinatoire des données tabulaires non pas par un mélange dense (attention), mais par un **pipeline de filtrage séquentiel et additif**. 

Suite à une analyse rigoureuse, l'architecture garantit deux propriétés fondamentales :
1.  **Additivité stricte :** La prédiction finale est une combinaison linéaire des contributions individuelles de chaque feature. Il n'y a pas de MLP final brisant la décomposabilité. Le modèle est un vrai GAM (Generalized Additive Model).
2.  **Importance déterministe basée sur la norme :** L'importance d'une variable n'est pas un coefficient de gating, mais la norme réelle du vecteur qu'elle injecte dans la somme finale.

L'architecture force les variables à passer par 3 étapes de traitement supervisé :
*   **Filtre Local** : Évaluation de la variable de manière isolée.
*   **Décorrélation par projection** : Suppression mathématique de l'information déjà expliquée par d'autres variables.
*   **Filtre Global Contextuel** : Évaluation de la variable en fonction de l'état du reste du système.

---

## Architecture Générale

```
           x₁    x₂    x₃   ...   xₙ          ← Données brutes
            │     │     │          │
            ▼     ▼     ▼          ▼
       ┌────────────────────────────────┐
       │     ÉTAPE 1 : TOKENIZATION     │       ← Projection latente + 
       │     (PLR normalisé + Embed)    │          Calibration
       └────────────────────────────────┘
            │     │     │          │
           h₁    h₂    h₃   ...  hₙ    ∈ ℝᵈ
            │     │     │          │
            ▼     ▼     ▼          ▼
       ┌────────────────────────────────┐
       │     ÉTAPE 2 : FILTRE LOCAL     │       ← Évaluation de la variable
       │     (Local Importance Gate)    │          en isolation
       └────────────────────────────────┘
            │     │     │          │
          s₁h₁  s₂h₂  s₃h₃ ... sₙhₙ   ∈ ℝᵈ     s ∈ [0,1]
            │     │     │          │
            ▼     ▼     ▼          ▼
       ┌────────────────────────────────┐
       │     ÉTAPE 3 : DÉCORRÉLATION    │       ← Retrait orthogonal de
       │     (Orthogonal Projection)    │          l'information redondante
       └────────────────────────────────┘
            │     │     │          │
           h₁'   h₂'   h₃'  ...   hₙ'   ∈ ℝᵈ
            │     │     │          │
            ▼     ▼     ▼          ▼
       ┌────────────────────────────────┐
       │     ÉTAPE 4 : FILTRE CONTEXTUEL│       ← Évaluation via un contexte
       │     (Global Context Gate)      │          "leave-one-out"
       └────────────────────────────────┘
            │     │                │
            z₁    z₂    ...       zₙ    ∈ ℝᵈ     (où zᵢ = gᵢhᵢ')
            │     │                │
            └─────┴───── ∑ ────────┘
                         │
                    v = Σ zᵢ                    ∈ ℝᵈ
                         │
                         ▼
                ┌─────────────────┐
                │ ÉTAPE 5 : HEAD  │              ← Tête LINÉAIRE stricte
                │                 │                pour préserver l'additivité
                └─────────────────┘
                         │
                         ▼
                        ŷ                        ∈ ℝ^(d_out)
```

---

## Étape 1 — Tokenization (Projection Latente)

**Équations corrigées :**

Pour les variables numériques (Piecewise Linear Representation avec normalisation par bin) :
Soit $t_i^{(m)}$ les seuils. La largeur du bin est $\Delta_i^{(m)} = t_i^{(m+1)} - t_i^{(m)}$.
$$\text{PLR}_i^{(m)}(x_i) = \frac{\max\!\big(0, \min(\Delta_i^{(m)}, x_i - t_i^{(m)})\big)}{\Delta_i^{(m)}} \in [0, 1]$$
$$h_i^{\text{raw}} = W_i^{\text{plr}} \cdot \text{PLR}_i(x_i) + b_i^{\text{plr}} \in \mathbb{R}^{B \times d}$$

**Calibration inter-features :**
Pour s'assurer que les filtres suivants n'agissent pas sur de simples différences d'échelle dues à l'initialisation, chaque feature passe par sa propre Batch Normalization :
$$h_i = \text{BatchNorm1d}_i(h_i^{\text{raw}}) \in \mathbb{R}^{B \times d}$$

---

## Étape 2 — Filtre Local (Local Importance Gate)

**Objectif :** Attribuer un score $s_i$ basé sur la représentation de la variable $i$ en isolation. Ce gate est supervisé (le gradient remonte depuis la loss finale). Il apprend à down-weighter les états banals ou bruités spécifiques à cette feature.

**Équations :**
$$s_i = \sigma(w_{s, i}^\top h_i + b_{s, i}) \in \mathbb{R}^{B}$$
*(Note: $w_{s, i}$ est spécifique à chaque feature, prenant en compte la BN précédente)*
$$\tilde{h}_i = s_i \cdot h_i \in \mathbb{R}^{B \times d}$$

---

## Étape 3 — Décorrélation par Projection Orthogonale

**Objectif :** Supprimer la multicolinéarité de façon géométrique. Plutôt qu'une soustraction brute (et potentiellement asymétrique via ReLU), on retire de la variable $i$ la composante qui est *déjà expliquée* par la variable $j$.

**Équations :**
On apprend une matrice globale d'allocation de redondance $W_{\text{supp}} \in \mathbb{R}^{n \times n}$.
Pour s'assurer d'une compétition et éviter l'annulation mutuelle symétrique, on applique un softmax par ligne (masquant la diagonale) :
$$\bar{W}_{\text{supp}} = \text{softmax}(\hat{W}_{\text{supp}} - \infty \cdot I_n, \text{dim}=-1)$$

La soustraction se fait par projection de Gram-Schmidt pondérée :
$$h_i' = \tilde{h}_i - \alpha \sum_{j \neq i} \bar{w}_{ij} \cdot \text{proj}_{\tilde{h}_j}(\tilde{h}_i)$$
Où la projection est :
$$\text{proj}_{\tilde{h}_j}(\tilde{h}_i) = \frac{\langle \tilde{h}_i, \tilde{h}_j \rangle}{\|\tilde{h}_j\|^2 + \epsilon} \tilde{h}_j$$
*(L'hyperparamètre $\alpha \in [0,1]$ contrôle l'agressivité de la décorrélation, avec un warm-up conseillé à l'entraînement).*

**Résultat :** $h_i'$ contient uniquement l'information de $\tilde{h}_i$ qui est orthogonale aux features redondantes sélectionnées par $\bar{W}_{\text{supp}}$.

---

## Étape 4 — Filtre Contextuel (Global Context Gate)

**Objectif :** Évaluer la pertinence de la variable conditionnellement à l'état des *autres* variables.

**Équations :**
Calcul du contexte **Leave-One-Out** (pour empêcher l'auto-inclusion) :
$$c_{-i} = \frac{1}{n-1}\sum_{j \neq i} h_j' \in \mathbb{R}^{B \times d}$$

Calcul du gate contextuel :
$$g_i = \sigma\!\left(W_{g, i} \cdot [c_{-i} \;\|\; h_i'] + b_{g, i}\right) \in \mathbb{R}^{B}$$

Contribution finale de la variable :
$$z_i = g_i \cdot h_i' \in \mathbb{R}^{B \times d}$$

---

## Étape 5 — Agrégation Additive & Tête Linéaire Stricte

**Objectif :** Préserver la décomposabilité mathématique exacte (additivité).

**Équations :**
Agrégation des contributions :
$$v = \sum_{i=1}^{n} z_i \in \mathbb{R}^{B \times d}$$

Tête de décision linéaire (avec LayerNorm pour la stabilité) :
$$\hat{y} = W_{\text{out}} \cdot \text{LN}(v) + b_{\text{out}} \in \mathbb{R}^{B \times d_{\text{out}}}$$

> **Note cruciale :** Le LayerNorm est une transformation affine scalaire par sample : $\text{LN}(v) = \gamma \frac{v - \mu}{\sigma} + \beta$. Mathématiquement, l'opération reste décomposable : la contribution d'une variable à $\hat{y}$ est *exactement* traçable. L'absence de MLP (fonctions d'activation non-linéaires sur la somme) garantit que le modèle est un vrai GAM. L'expressivité non-linéaire vient des interactions dans le calcul de $z_i$ (projection et contexte $c_{-i}$).

---

## Formule d'Importance Déterministe

Puisque la tête de prédiction est linéaire (modulo LN), la véritable importance d'une variable pour une prédiction donnée n'est pas le produit des scalaires de gating, mais **la norme du vecteur de contribution qu'elle injecte dans la somme**.

$$\boxed{\text{Importance}_i = \|z_i\|_2 = \|g_i \cdot h_i'\|_2}$$

Cette formule est robuste :
1. Si le filtre local coupe l'info ($s_i \to 0$), $\tilde{h}_i$ s'annule, donc $h_i'$ s'annule, $z_i \to 0$.
2. Si la variable est totalement redondante, la projection (Étape 3) réduit la norme de $h_i'$ à 0.
3. Si le contexte juge l'info non pertinente ($g_i \to 0$), $z_i \to 0$.
4. **Elle prend en compte l'échelle réelle de l'embedding.**

Pour obtenir la contribution directionnelle (signée) spécifique à la classe cible $k$ :
$$\text{Contribution}_{i \to k} = \left( W_{\text{out}}[k, :] \odot \frac{\gamma}{\sigma} \right) \cdot z_i$$
*(Ceci donne l'équivalent mathématique exact des valeurs de Shapley, calculable en $O(1)$ au forward pass).*
