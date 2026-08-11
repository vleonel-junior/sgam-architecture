import torch
import torch.nn as nn
import math
from typing import List, Optional

class NumericalPLRTokenizer(nn.Module):
    """
    Étape 1 (Numérique) : Tokenization par Piecewise Linear Representation (PLR).
    """
    def __init__(self, n_num_features: int, d_token: int, n_bins: int = 8):
        super().__init__()
        self.n_num_features = n_num_features
        self.d_token = d_token
        self.n_bins = n_bins
        
        # Limites des bins (t_i^{(m)}). Shape: (n_num_features, n_bins + 1)
        boundaries = torch.linspace(-3, 3, n_bins + 1).unsqueeze(0).repeat(n_num_features, 1)
        self.boundaries = nn.Parameter(boundaries)
        
        # Projection linéaire: W_plr * PLR(x) + b_plr
        self.weight = nn.Parameter(torch.Tensor(n_num_features, d_token, n_bins))
        self.bias = nn.Parameter(torch.Tensor(n_num_features, d_token))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    @torch.no_grad()
    def set_thresholds_from_data(self, x_num: torch.Tensor):
        """
        Recalibre les seuils des bins PLR sur les quantiles empiriques de x_num: (N, n_num_features).
        """
        qs = torch.linspace(0.0, 1.0, self.n_bins + 1, device=x_num.device, dtype=x_num.dtype)
        new_boundaries = torch.stack(
            [torch.quantile(x_num[:, i], qs) for i in range(self.n_num_features)]
        )
        # Garantit des bins strictement non dégénérés même si quantiles égaux
        bump = torch.arange(self.n_bins + 1, device=x_num.device, dtype=x_num.dtype) * 1e-6
        new_boundaries = new_boundaries + bump
        self.boundaries.copy_(new_boundaries.to(self.boundaries.dtype))

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        """
        x_num: (B, n_num_features)
        Returns:
        h: (B, n_num_features, d_token)
        """
        B = x_num.shape[0]
        x_expanded = x_num.unsqueeze(-1)
        
        t_m = self.boundaries[:, :-1].unsqueeze(0)
        t_m_plus_1 = self.boundaries[:, 1:].unsqueeze(0)
        
        delta = (t_m_plus_1 - t_m).clamp_min(1e-8)
        
        x_centered = x_expanded - t_m
        plr_vals = torch.clamp(x_centered, min=0.0)
        plr_vals = torch.min(plr_vals, delta)
        plr_vals = plr_vals / delta
        
        h = torch.einsum("bfk,fdk->bfd", plr_vals, self.weight) + self.bias.unsqueeze(0)
        return h

class CategoricalTokenizer(nn.Module):
    """
    Étape 1 (Catégorielle) : Embeddings pour variables catégorielles.
    """
    def __init__(self, cardinalities: List[int], d_token: int):
        super().__init__()
        self.cardinalities = cardinalities
        self.d_token = d_token
        
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, d_token) for card in cardinalities
        ])
        
    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """
        x_cat: (B, n_cat_features) - Tenseur d'entiers (indices des catégories)
        Returns:
        h_cat: (B, n_cat_features, d_token)
        """
        # Embed chaque colonne séparément
        embedded = [
            emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)
        ] # Liste de (B, d_token)
        
        h_cat = torch.stack(embedded, dim=1) # (B, n_cat_features, d_token)
        return h_cat

class FeatureTokenizer(nn.Module):
    """
    Combine les tokenizers numérique (PLR) et catégoriel (Embedding)
    suivis d'une calibration BatchNorm1d unifiée par feature (Étape 1 complète).
    """
    def __init__(self, n_num_features: int = 0, categories: Optional[List[int]] = None, d_token: int = 16, n_bins: int = 8):
        super().__init__()
        self.n_num_features = n_num_features
        self.categories = categories or []
        self.n_cat_features = len(self.categories)
        self.n_total_features = self.n_num_features + self.n_cat_features
        self.d_token = d_token
        
        assert self.n_total_features > 0, "Le modèle doit avoir au moins une feature (numérique ou catégorielle)."
        
        if self.n_num_features > 0:
            self.num_tokenizer = NumericalPLRTokenizer(n_num_features, d_token, n_bins)
        else:
            self.num_tokenizer = None
            
        if self.n_cat_features > 0:
            self.cat_tokenizer = CategoricalTokenizer(self.categories, d_token)
        else:
            self.cat_tokenizer = None
            
        # Calibration inter-features globale
        self.bn = nn.BatchNorm1d(self.n_total_features * d_token)
        
    def forward(self, x_num: Optional[torch.Tensor] = None, x_cat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_num: (B, n_num_features) optionnel
        x_cat: (B, n_cat_features) optionnel
        Returns:
        h_calibrated: (B, n_total_features, d_token)
        """
        tokens = []
        
        if self.num_tokenizer is not None:
            assert x_num is not None, "x_num est requis car n_num_features > 0"
            h_num = self.num_tokenizer(x_num)
            tokens.append(h_num)
            
        if self.cat_tokenizer is not None:
            assert x_cat is not None, "x_cat est requis car des catégories ont été spécifiées"
            h_cat = self.cat_tokenizer(x_cat)
            tokens.append(h_cat)
            
        # Concaténation des tokens le long de la dimension des features (dim=1)
        h_raw = torch.cat(tokens, dim=1) # (B, n_total_features, d_token)
        
        B = h_raw.shape[0]
        h_flat = h_raw.reshape(B, -1)
        h_bn = self.bn(h_flat)
        h_calibrated = h_bn.reshape(B, self.n_total_features, self.d_token)
        
        return h_calibrated

