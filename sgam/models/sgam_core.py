import torch
import torch.nn as nn
from typing import List, Optional

from sgam.modules.tokenization import FeatureTokenizer
from sgam.modules.gating import LocalImportanceGate, GlobalContextGate
from sgam.modules.decorrelation import AsymmetricOrthogonalDecorrelation

class RMSNorm(nn.Module):
    """
    Implémentation custom de RMSNorm pour rétrocompatibilité avec PyTorch < 2.4.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d)
        # rms: (B, 1)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms) + self.bias
        
    def get_rms(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne la valeur RMS pour usage dans la formule de contribution."""
        return torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)

class SGAM(nn.Module):
    def __init__(self, n_num_features: int = 0, categories: Optional[List[int]] = None, d_token: int = 16, n_bins: int = 8, d_out: int = 1):
        super().__init__()
        self.n_num_features = n_num_features
        self.categories = categories or []
        self.n_cat_features = len(self.categories)
        self.n_total_features = self.n_num_features + self.n_cat_features
        self.d_token = d_token
        self.d_out = d_out
        
        # Étapes 1 à 4
        self.tokenizer = FeatureTokenizer(n_num_features, categories, d_token, n_bins)
        self.local_gate = LocalImportanceGate(self.n_total_features, d_token)
        self.decorrelation = AsymmetricOrthogonalDecorrelation(self.n_total_features)
        self.context_gate = GlobalContextGate(self.n_total_features, d_token)
        
        # Étape 5 : Agrégation et Tête Linéaire
        self.norm = RMSNorm(d_token)
        self.head = nn.Linear(d_token, d_out, bias=True)
        
    def forward(self, x_num: Optional[torch.Tensor] = None, x_cat: Optional[torch.Tensor] = None, return_z: bool = False):
        """
        x_num: (B, n_num_features) optionnel
        x_cat: (B, n_cat_features) optionnel (indices entiers)
        """
        # Étape 1 : Tokenization + Calibration BatchNorm
        h = self.tokenizer(x_num=x_num, x_cat=x_cat)
        
        # Étape 2 : Filtre Local
        tilde_h = self.local_gate(h)
        
        # Étape 3 : Décorrélation Orthogonale
        h_prime = self.decorrelation(tilde_h)
        
        # Étape 4 : Filtre Contextuel (Leave-One-Out)
        z = self.context_gate(h_prime)
        
        # Étape 5 : Agrégation Additive + RMSNorm + Tête Linéaire
        v = torch.sum(z, dim=1) # (B, d_token)
        v_norm = self.norm(v) # (B, d_token)
        y_hat = self.head(v_norm) # (B, d_out)
        
        if return_z:
            return y_hat, z, v
        return y_hat

    def get_importance_scores(self, x_num: Optional[torch.Tensor] = None, x_cat: Optional[torch.Tensor] = None):
        """
        Calcule l'attribution exacte a posteriori et la magnitude L2 de chaque feature.
        """
        y_hat, z, v = self.forward(x_num=x_num, x_cat=x_cat, return_z=True)
        
        # 1. Magnitude absolue d'importance (Norme L2)
        importance_l2 = torch.norm(z, p=2, dim=-1) # (B, n_total_features)
        
        # 2. Attribution signée exacte
        rms_v = self.norm.get_rms(v) # (B, 1)
        scale_factor = self.norm.weight.unsqueeze(0) / rms_v # (B, d_token)
        
        baseline = torch.matmul(self.norm.bias, self.head.weight.T) + self.head.bias
        
        scaled_W = self.head.weight.unsqueeze(0) * scale_factor.unsqueeze(1) # (B, d_out, d_token)
        contributions = torch.einsum("bkd,bid->bik", scaled_W, z) # (B, n_total_features, d_out)
        
        return {
            "y_hat": y_hat,
            "baseline": baseline, # (d_out,)
            "contributions": contributions, # (B, n_total_features, d_out)
            "importance_l2": importance_l2 # (B, n_total_features)
        }

