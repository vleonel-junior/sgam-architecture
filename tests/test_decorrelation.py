import torch
from sgam.modules.decorrelation import AsymmetricOrthogonalDecorrelation

def test_perfect_duplication():
    """Scénario 1 : Duplication parfaite"""
    print("--- Test 1 : Duplication Parfaite ---")
    torch.manual_seed(42)
    B, N, d = 2, 2, 8
    model = AsymmetricOrthogonalDecorrelation(N)
    
    # Feature 0 et Feature 1 sont strictement identiques
    u = torch.randn(B, 1, d)
    h = u.repeat(1, 2, 1) # (B, 2, d)
    
    # Forward
    h_out = model(h)
    
    # Vérifications
    print(f"Norme initiale : {torch.norm(h[0,0]).item():.4f}")
    print(f"Norme sortie F0 : {torch.norm(h_out[0,0]).item():.4f}")
    print(f"Norme sortie F1 : {torch.norm(h_out[0,1]).item():.4f}")
    
    # On vérifie que la sortie n'est pas nulle (pas de collapse symétrique vers 0)
    assert torch.norm(h_out) > 1e-4, "Collapse symétrique détecté !"
    
    # On vérifie le sens (clip directionnel)
    dot_prod = torch.sum(h * h_out, dim=-1)
    assert (dot_prod >= 0).all(), "Retournement de signe détecté !"
    print("Test 1 OK : Pas de collapse total ni de retournement.\n")

def test_orthogonal_features():
    """Scénario 3 : Variables indépendantes (orthogonales)"""
    print("--- Test 3 : Features Orthogonales ---")
    B, N, d = 1, 2, 2
    model = AsymmetricOrthogonalDecorrelation(N)
    
    # Deux vecteurs orthogonaux
    u1 = torch.tensor([[[1.0, 0.0]]])
    u2 = torch.tensor([[[0.0, 1.0]]])
    h = torch.cat([u1, u2], dim=1) # (1, 2, 2)
    
    h_out = model(h)
    
    # La décorrélation d'éléments orthogonaux ne doit rien changer
    diff = torch.norm(h_out - h)
    print(f"Différence ||h_out - h|| = {diff.item():.6f}")
    assert diff < 1e-4, "Des vecteurs orthogonaux ont été modifiés !"
    print("Test 3 OK : L'orthogonalité préserve les features.\n")
    
if __name__ == "__main__":
    test_perfect_duplication()
    test_orthogonal_features()

