import torch
import torch.nn as nn
import math

class LocalImportanceGate(nn.Module):
    """
    Étape 2 : Filtre Local.
    Évalue la variable de manière isolée pour attribuer un score s_i.
    """
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        
        # Poids spécifiques à chaque feature
        self.w_s = nn.Parameter(torch.Tensor(n_features, d_token))
        self.b_s = nn.Parameter(torch.Tensor(n_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_s, a=math.sqrt(5))
        fan_in = self.d_token
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b_s, -bound, bound)
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, n_features, d_token)
        Returns:
        tilde_h: (B, n_features, d_token)
        """
        # Calcul du score s_i pour chaque feature
        # (B, F, D) * (F, D) -> (B, F)
        logits = torch.einsum("bfd,fd->bf", h, self.w_s) + self.b_s.unsqueeze(0)
        s = torch.sigmoid(logits)
        
        tilde_h = s.unsqueeze(-1) * h
        return tilde_h

class GlobalContextGate(nn.Module):
    """
    Étape 4 : Filtre Contextuel.
    Évalue la pertinence de la variable conditionnellement à l'état des autres variables (leave-one-out).
    """
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        
        # Le vecteur d'entrée fait 2 * d_token de large ([c_minus_i || h_i_prime])
        self.w_g = nn.Parameter(torch.Tensor(n_features, 2 * d_token))
        self.b_g = nn.Parameter(torch.Tensor(n_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.w_g, a=math.sqrt(5))
        fan_in = 2 * self.d_token
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.b_g, -bound, bound)
        
    def forward(self, h_prime: torch.Tensor) -> torch.Tensor:
        """
        h_prime: (B, n_features, d_token) - après décorrélation
        Returns:
        z: (B, n_features, d_token) - contribution finale
        """
        # 1. Calcul du contexte leave-one-out
        sum_h = torch.sum(h_prime, dim=1, keepdim=True) # (B, 1, d_token)
        c_minus_i = (sum_h - h_prime) / max(1, self.n_features - 1) # (B, F, D)
        
        # 2. Concaténation
        concat_input = torch.cat([c_minus_i, h_prime], dim=-1) # (B, F, 2D)
        
        # 3. Calcul du gate
        logits = torch.einsum("bfd,fd->bf", concat_input, self.w_g) + self.b_g.unsqueeze(0)
        g = torch.sigmoid(logits) # (B, F)
        
        # 4. Pondération finale
        z = g.unsqueeze(-1) * h_prime
        return z

