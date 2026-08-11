# SGAM : Sequential Gated Additive Model

*Filtrage Séquentiel et Résolution de la Multicolinéarité pour Données Tabulaires*

---

## Philosophie : Attribution Additive Exacte et Filtrage Supervisé

Le SGAM est conçu pour résoudre l'explosion combinatoire des données tabulaires non pas par un mélange dense (attention), mais par un **pipeline de filtrage séquentiel**. 

Contrairement à un modèle purement additif naïf (GAM classique) où les variables sont indépendantes, SGAM permet aux variables d'interagir dynamiquement (via la suppression de redondance et le contexte global) **avant** de les agréger. Cela lui confère une grande expressivité.

Cependant, l'architecture garantit une propriété d'interprétabilité fondamentale : **l'attribution additive exacte a posteriori**. Pour toute prédiction *donnée*, la sortie du modèle se décompose en une somme exacte et déterministe des contributions de chaque feature, satisfaisant les axiomes d'efficacité et de symétrie (comme les valeurs de Shapley), mais avec un coût computationnel en $O(1)$.

L'architecture force les variables à passer par 3 étapes de traitement supervisé :
*   **Filtre Local** : Évaluation de la variable de manière isolée.
*   **Décorrélation par projection** : Suppression mathématique de l'information déjà expliquée par d'autres variables (avec priorité asymétrique).
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
       │     ÉTAPE 3 : DÉCORRÉLATION    │       ← Retrait orthogonal avec
       │     (Orthogonal Projection)    │          priorité asymétrique
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
                │  (avec RMSNorm) │                pour préserver l'additivité
                └─────────────────┘
                         │
                         ▼
                        ŷ                        ∈ ℝ^(d_out)
```

---

## Étape 1 — Tokenization (Projection Latente)

**Équations :**

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

**Objectif :** Supprimer la multicolinéarité de façon géométrique sans risquer l'annulation mutuelle (collapse symétrique).

**Équations :**
On définit des gates de suppression **indépendants**, asymétrisés par la norme des vecteurs pour imposer une priorité (le vecteur le plus "fort" inhibe le plus faible, pas l'inverse) :
$$\rho_i = \|\tilde{h}_i\|_2$$
$$\bar{w}_{ij} = \sigma(\hat{w}_{ij}) \cdot \sigma\big(\tau(\rho_j - \rho_i)\big) \in [0, 1]$$

L'intensité globale d'inhibition est strictement bornée :
$$\alpha = \sigma(\hat{\alpha}) \in [0, 1]$$

La soustraction se fait par projection de Gram-Schmidt pondérée :
$$h_i^{\text{temp}} = \tilde{h}_i - \alpha \sum_{j \neq i} \bar{w}_{ij} \cdot \text{proj}_{\tilde{h}_j}(\tilde{h}_i)$$
Où la projection est :
$$\text{proj}_{\tilde{h}_j}(\tilde{h}_i) = \frac{\langle \tilde{h}_i, \tilde{h}_j \rangle}{\|\tilde{h}_j\|^2 + \epsilon} \tilde{h}_j$$

**Sécurité (Anti-Overshoot et Directionnelle) :**
Le clip de norme protège la magnitude ($\|h_i'\| \le \|\tilde{h}_i\|$), mais ne protège pas contre un retournement complet de signe si les suppressions s'accumulent (ex: $h_i^{\text{temp}} = -\tilde{h}_i$). Pour garantir la fidélité, on applique d'abord un clip directionnel (on met à 0 si le vecteur s'est retourné), puis le clip de norme :
$$h_i^{\text{dir}} = h_i^{\text{temp}} \text{ si } \langle h_i^{\text{temp}}, \tilde{h}_i \rangle \ge 0, \text{ sinon } 0$$
$$h_i' = h_i^{\text{dir}} \cdot \min\left(1, \frac{\|\tilde{h}_i\|_2}{\|h_i^{\text{dir}}\|_2 + \epsilon}\right)$$

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

## Étape 5 — Agrégation Additive & Tête Linéaire (RMSNorm)

**Objectif :** Préserver la décomposabilité mathématique exacte. L'utilisation de RMSNorm (Root Mean Square Normalization) au lieu de LayerNorm est cruciale : RMSNorm ne soustrait pas la moyenne, évitant ainsi le problème de recentrage croisé qui briserait l'additivité exacte.

**Équations :**
Agrégation des contributions :
$$v = \sum_{i=1}^{n} z_i \in \mathbb{R}^{B \times d}$$

RMSNorm (où $\text{rms}(v) = \sqrt{\frac{1}{d}\sum v_j^2 + \epsilon}$) :
$$v_{\text{norm}} = \gamma \odot \frac{v}{\text{rms}(v)} + \beta$$

Tête de décision linéaire :
$$\hat{y} = W_{\text{out}} \cdot v_{\text{norm}} + b_{\text{out}} \in \mathbb{R}^{B \times d_{\text{out}}}$$

---

## Formule d'Importance et d'Attribution

Puisque la tête de prédiction est linéaire (modulo le scaling par RMSNorm), l'attribution de chaque variable $x_i$ à la classe cible $k$ est déterministe et s'écrit analytiquement.

Pour la classe ciblée $k$ :

$$\hat{y}_k = \sum_{i=1}^n \mathrm{Contrib}_{i \to k} + \mathrm{Baseline}_k$$

Avec :

$$\mathrm{Contrib}_{i \to k} = \left( W_{\mathrm{out}}[k, :] \odot \frac{\gamma}{\mathrm{rms}(v)} \right) \cdot z_i$$

$$\mathrm{Baseline}_k = W_{\mathrm{out}}[k, :] \cdot \beta + b_{\mathrm{out}, k}$$

Cette formulation garantit l'axiome d'efficacité de Shapley (la somme des contributions égale exactement la sortie) ainsi qu'une propriété de cohérence interne analogue à la symétrie (des contributions identiques produisent des attributions identiques), sans toutefois définir un jeu coopératif formel sur les coalitions de features.

*Note de rigueur :* Il s'agit d'une attribution a posteriori (analogue en esprit à LRP), et non d'une suppression contrefactuelle. Mettre manuellement $z_i = 0$ recalculerait le dénominateur $\mathrm{rms}(v)$, modifiant la prédiction d'une valeur légèrement différente de la contribution isolée calculée ici.

La magnitude globale d'importance d'une variable se mesure simplement par :

$$\mathrm{Importance}_i = \|z_i\|_2$$

