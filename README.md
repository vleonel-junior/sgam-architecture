# SGAM : Sequential Gated Additive Model

*Filtrage Séquentiel et Résolution de la Multicolinéarité pour Données Tabulaires*

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

SGAM est une nouvelle architecture de réseau de neurones dédiée aux données tabulaires. Elle résout la dépendance combinatoire entre variables non pas par des mécanismes d'attention denses coûteux et opaques, mais par un **pipeline de filtrage séquentiel supervisé**.

SGAM garantit une propriété d'interprétabilité fondamentale : **l'attribution additive exacte a posteriori**. La sortie du modèle se décompose de façon déterministe en une somme de contributions individuelles par feature, avec un coût computationnel en O(1).

---

## 🌟 Innovations Principales

1. **Attribution Additive Exacte a posteriori** :
   Grâce à une tête de décision linéaire couplée à la normalisation **RMSNorm** (RMSNorm(v) = gamma * v / rms(v) + beta), la prédiction y_hat_k pour chaque classe k se décompose exactement en :
   y_hat_k = sum_i Contrib_{i -> k} + Baseline_k
   L'axiome d'efficacité de Shapley (sum Contrib_i + Baseline = y_hat) est garanti au bit près (erreur < 1e-7).

2. **Décorrélation Orthogonale Asymétrique (Étape 3)** :
   Suppression géométrique de la multicolinéarité via des projections de Gram-Schmidt pondérées. Les représentations de plus forte norme ||h_tilde_i|| inhibent les variables plus faibles de façon asymétrique.
   - **Clip Directionnel** : <h_i_temp, h_tilde_i> >= 0 pour interdire tout retournement de signe lors d'inhibitions cumulées.
   - **Clip de Norme** : ||h_i'|| <= ||h_tilde_i|| pour prévenir l'overshoot d'amplitude.

3. **Filtrage Contextuel Leave-One-Out (Étape 4)** :
   Calcul du contexte c_{-i} = 1/(n-1) * sum_{j != i} h'_j pour moduler l'importance d'une variable par rapport au reste du système sans auto-inclusion.

4. **Support Natif des Variables Numériques & Catégorielles** :
   - Numériques : Tokenization PLR (Piecewise Linear Representation) avec limites de bins ajustables sur les quantiles empiriques réels (set_thresholds_from_data).
   - Catégorielles : Embeddings dédiés par modalité (nn.Embedding).
   - Calibration unifiée inter-features via BatchNorm1d.

---

## 📁 Structure du Projet

`	ext
sgam-architecture/
├── SGAM_ARCHITECTURE.md        # Spécifications et preuves mathématiques complètes
├── sgam/                       # Package Python du modèle
│   ├── modules/
│   │   ├── tokenization.py     # PLR + Categorical Embedding + BatchNorm1d
│   │   ├── gating.py           # LocalImportanceGate & GlobalContextGate
│   │   └── decorrelation.py    # Gram-Schmidt Asymétrique + Clips directionnel & norme
│   └── models/
│       └── sgam_core.py        # Modèle SGAM complet & extraction d'attribution
├── benchmark/                  # Framework d'expérimentation et de benchmark tabulaire
│   ├── bin/
│   │   ├── sgam.py             # Script d'entraînement SGAM sur jeux de données
│   │   ├── mlp.py, tune.py...  # Baselines et utilitaires de benchmark
│   ├── lib/                    # Loaders de données et métriques
├── tests/                      # Suite de tests unitaires et d'intégration
│   ├── test_decorrelation.py   # Test synthétique de suppression de redondance
│   └── test_sgam_core.py       # Validation de l'axiome d'efficacité & duplication
`

---

## 🚀 Utilisation Rapide (PyTorch)

`python
import torch
from sgam.models.sgam_core import SGAM

# Initialisation du modèle pour 4 num et 2 cat (cardinalités 5 et 10)
model = SGAM(
    n_num_features=4,
    categories=[5, 10],
    d_token=16,
    d_out=1
)

# Entrées factices
x_num = torch.randn(8, 4)
x_cat = torch.tensor([[0, 5], [4, 9], [2, 3], [1, 0], [3, 2], [1, 1], [0, 4], [2, 8]])

# 1. Forward pass classique
y_hat = model(x_num=x_num, x_cat=x_cat)

# 2. Explication exacte & attribution des contributions
scores = model.get_importance_scores(x_num=x_num, x_cat=x_cat)
print('Prédiction :', scores['y_hat'])
print('Contributions par feature :', scores['contributions'].shape) # (8, 6, 1)
print('Importance L2 :', scores['importance_l2'].shape) # (8, 6)
`

---

## 🧪 Lancer les Tests

Pour exécuter la suite de tests unitaires et vérifier les propriétés mathématiques :

`ash
python tests/test_decorrelation.py
python tests/test_sgam_core.py
`

---

## 📜 Citation & Références

- **Gram-Schmidt & Redondance** : Zaheer et al., 2017 (DeepSets); Wagstaff et al., 2019.
- **PLR Tokenization** : Gorishniy et al., 2022 (On Embeddings for Numerical Features in Tabular Deep Learning).
- **RMSNorm** : Zhang & Sennrich, 2019 (Root Mean Square Layer Normalization).
