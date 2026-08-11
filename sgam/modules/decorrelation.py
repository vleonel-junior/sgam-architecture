import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricOrthogonalDecorrelation(nn.Module):
    """
    Étape 3 : Décorrélation par projection orthogonale asymétrique.
    Supprime la multicolinéarité de façon géométrique sans risquer l'annulation mutuelle
    et avec un garde-fou contre le retournement de direction (anti-overshoot).
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        
        # W_hat[i, j] = Logit d'interaction brute de j sur i
        # L'initialisation proche de 0 donne des sigmoid(W_hat) proches de 0.5 au départ.
        self.W_hat = nn.Parameter(torch.zeros(n_features, n_features))
        
        # Température tau pour l'asymétrie basée sur la norme
        self.tau = nn.Parameter(torch.ones(1))
        
        # Intensité globale de soustraction alpha (bornée dans [0, 1])
        # Initialisé à -3.0 (alpha ~ 0.05 au démarrage pour servir de warm-up progressif)
        self.alpha_logit = nn.Parameter(torch.tensor(-3.0))
        
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, N, d) Représentations des features après le filtre local
        Returns:
        h_out: (B, N, d) Représentations décorrélées
        """
        B, N, d = h.shape
        
        # 1. Calcul des priorités asymétriques basées sur la norme
        # rho: (B, N)
        rho = torch.norm(h, p=2, dim=-1)
        
        # rho_diff[b, i, j] = rho[b, j] - rho[b, i]
        # On veut que si rho_j > rho_i, le sigmoïde soit > 0.5 (j inhibe i)
        rho_i = rho.unsqueeze(2) # (B, N, 1)
        rho_j = rho.unsqueeze(1) # (B, 1, N)
        rho_diff = rho_j - rho_i # (B, N, N)
        
        # Priorité asymétrique: (B, N, N)
        priority_gate = torch.sigmoid(self.tau * rho_diff)
        
        # 2. Calcul des poids de suppression (gates)
        # base_gate: (1, N, N) broadcasté à (B, N, N)
        base_gate = torch.sigmoid(self.W_hat.unsqueeze(0))
        
        # W_bar[b, i, j]: Poids avec lequel j va inhiber i
        W_bar = base_gate * priority_gate
        
        # On annule la diagonale pour qu'une feature ne s'auto-inhibe pas
        mask = 1.0 - torch.eye(N, device=h.device).unsqueeze(0)
        W_bar = W_bar * mask
        
        # 3. Calcul de la soustraction par projection de Gram-Schmidt
        # dot_ij[b, i, j] = <h_i, h_j>
        dot_ij = torch.einsum("bid,bjd->bij", h, h)
        
        # norm_j_sq[b, 1, j] = ||h_j||^2
        norm_j_sq = torch.sum(h**2, dim=-1).unsqueeze(1)
        
        # proj_coef[b, i, j] = <h_i, h_j> / (||h_j||^2 + eps)
        proj_coef = dot_ij / (norm_j_sq + 1e-6)
        
        # sum_j_term[b, i, d] = sum_j (W_bar[b, i, j] * proj_coef[b, i, j] * h[b, j, d])
        # C'est la somme pondérée des projections de h_i sur tous les h_j
        sum_j_term = torch.einsum("bij,bij,bjd->bid", W_bar, proj_coef, h)
        
        # Alpha borné dans [0, 1]
        alpha = torch.sigmoid(self.alpha_logit)
        
        # h_temp: Soustraction
        h_temp = h - alpha * sum_j_term
        
        # 4. Sécurité Directionnelle (Clip directionnel)
        # On vérifie que le vecteur n'a pas été retourné à 180°
        # dot_temp_orig[b, i] = <h_temp_i, h_i>
        dot_temp_orig = torch.einsum("bid,bid->bi", h_temp, h)
        # mask_dir = 1 si le sens est préservé ou vecteur nul, 0 sinon
        mask_dir = (dot_temp_orig >= 0).float().unsqueeze(-1)
        
        h_dir = h_temp * mask_dir
        
        # 5. Sécurité de Norme (Clip d'amplitude)
        # ||h_out||_2 <= ||h_orig||_2
        norm_orig = torch.norm(h, p=2, dim=-1, keepdim=True)
        norm_dir = torch.norm(h_dir, p=2, dim=-1, keepdim=True)
        
        ratio = norm_orig / (norm_dir + 1e-6)
        clip_coef = torch.clamp(ratio, max=1.0)
        
        h_out = h_dir * clip_coef
        
        return h_out

