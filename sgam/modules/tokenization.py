import torch
import torch.nn as nn
import math

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
        # Initialisés uniformément entre -3 et 3 (en supposant des données standardisées)
        boundaries = torch.linspace(-3, 3, n_bins + 1).unsqueeze(0).repeat(n_num_features, 1)
        self.boundaries = nn.Parameter(boundaries)
        
        # Projection linéaire: W_plr * PLR(x) + b_plr
        # On utilise une couche linéaire par feature: (n_features, n_bins) -> (n_features, d_token)
        self.weight = nn.Parameter(torch.Tensor(n_num_features, d_token, n_bins))
        self.bias = nn.Parameter(torch.Tensor(n_num_features, d_token))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x_num: torch.Tensor) -> torch.Tensor:
        """
        x_num: (B, n_num_features)
        Returns:
        h: (B, n_num_features, d_token)
        """
        B = x_num.shape[0]
        # x_expanded: (B, n_num_features, 1)
        x_expanded = x_num.unsqueeze(-1)
        
        # t_m: (1, n_num_features, n_bins)
        t_m = self.boundaries[:, :-1].unsqueeze(0)
        # t_m_plus_1: (1, n_num_features, n_bins)
        t_m_plus_1 = self.boundaries[:, 1:].unsqueeze(0)
        
        delta = t_m_plus_1 - t_m # (1, n_num_features, n_bins)
        
        # PLR = max(0, min(delta, x - t_m)) / delta
        # Shape: (B, n_num_features, n_bins)
        x_centered = x_expanded - t_m
        plr_vals = torch.clamp(x_centered, min=0.0)
        plr_vals = torch.min(plr_vals, delta)
        plr_vals = plr_vals / (delta + 1e-6)
        
        # Projection W_plr * PLR + b_plr
        # plr_vals: (B, n_features, n_bins)
        # weight: (n_features, d_token, n_bins)
        # On fait le produit matrice-vecteur pour chaque feature
        # (B, F, Bins) * (F, D, Bins) -> (B, F, D)
        h = torch.einsum("bfk,fdk->bfd", plr_vals, self.weight) + self.bias.unsqueeze(0)
        return h

class FeatureTokenizer(nn.Module):
    """
    Combine le tokenizer numérique et la BatchNorm par feature (Étape 1 complète).
    Note : Le support catégoriel pur (Embedding) pourra être ajouté ici.
    """
    def __init__(self, n_features: int, d_token: int, n_bins: int = 8):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        
        self.num_tokenizer = NumericalPLRTokenizer(n_features, d_token, n_bins)
        
        # Calibration inter-features : BatchNorm1d(n_features * d_token)
        # Cela permet d'avoir mu et sigma indépendants pour chaque feature et chaque dimension.
        self.bn = nn.BatchNorm1d(n_features * d_token)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, n_features)
        Returns:
        h_calibrated: (B, n_features, d_token)
        """
        B = x.shape[0]
        # 1. Projection latente
        h_raw = self.num_tokenizer(x) # (B, F, D)
        
        # 2. Calibration BatchNorm1d
        # Reshape pour que BatchNorm1d voit (B, F*D)
        h_flat = h_raw.reshape(B, -1)
        h_bn = self.bn(h_flat)
        h_calibrated = h_bn.reshape(B, self.n_features, self.d_token)
        
        return h_calibrated

