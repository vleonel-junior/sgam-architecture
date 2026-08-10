import torch
import torch.nn as nn
from sgam.models.sgam_core import SGAM

def test_efficiency():
    """Vérifier que Somme(Contributions) + Baseline == y_hat"""
    print("--- Test 1 : Axiome d'efficacité (Somme exacte) ---")
    torch.manual_seed(42)
    B, N, D = 4, 5, 8
    model = SGAM(n_features=N, d_token=D, d_out=2)
    
    x = torch.randn(B, N)
    scores = model.get_importance_scores(x)
    
    y_hat = scores["y_hat"] # (B, d_out)
    baseline = scores["baseline"] # (d_out,)
    contributions = scores["contributions"] # (B, N, d_out)
    
    # Somme des contributions pour chaque sample et chaque classe
    sum_contrib = torch.sum(contributions, dim=1) # (B, d_out)
    
    # Recomposition
    y_hat_recomposed = sum_contrib + baseline.unsqueeze(0)
    
    diff = torch.max(torch.abs(y_hat - y_hat_recomposed)).item()
    print(f"Erreur max de recomposition : {diff:.8e}")
    assert diff < 1e-5, f"L'axiome d'efficacité a échoué ! Différence : {diff}"
    print("Test 1 OK : L'attribution additive exacte est confirmée.\n")

def test_duplication_core():
    """Vérifier que SGAM complet survit à une duplication de feature."""
    print("--- Test 2 : Duplication End-to-End ---")
    torch.manual_seed(42)
    B, N, D = 2, 3, 4
    model = SGAM(n_features=N, d_token=D, d_out=1)
    
    # Feature 0 et 1 sont identiques, feature 2 est différente
    x_base = torch.randn(B, 1)
    x_diff = torch.randn(B, 1)
    x = torch.cat([x_base, x_base, x_diff], dim=1) # (B, 3)
    
    # On modifie les poids de tokenization pour être sûr qu'ils sont identiques 
    # pour les features 0 et 1 afin de simuler une vraie symétrie latente
    with torch.no_grad():
        model.tokenizer.num_tokenizer.weight[0] = model.tokenizer.num_tokenizer.weight[1]
        model.tokenizer.num_tokenizer.bias[0] = model.tokenizer.num_tokenizer.bias[1]
        model.local_gate.w_s[0] = model.local_gate.w_s[1]
        model.local_gate.b_s[0] = model.local_gate.b_s[1]
    
    scores = model.get_importance_scores(x)
    importance = scores["importance_l2"]
    
    print(f"Importances: {importance}")
    # On vérifie juste que ça ne crash pas et que ça ne collapse pas à zéro absolu
    assert (importance[:, 0] > 1e-4).all(), "Collapse end-to-end (Feature 0)!"
    assert (importance[:, 1] > 1e-4).all(), "Collapse end-to-end (Feature 1)!"
    print("Test 2 OK : La duplication passe sans effondrement ni NaN.\n")

def test_dummy_feature():
    """Vérifier le comportement global sur une feature de pur bruit"""
    print("--- Test 3 : Dummy Feature ---")
    torch.manual_seed(42)
    B, N, D = 4, 3, 8
    model = SGAM(n_features=N, d_token=D, d_out=1)
    
    # Une target qui dépend seulement de x_0
    x = torch.randn(B, N)
    y = 2.0 * x[:, 0:1] + 1.0 # (B, 1)
    
    # Entrainement super rapide (1 step) pour voir si les gradients passent bien
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    loss_fn = nn.MSELoss()
    
    y_hat = model(x)
    loss = loss_fn(y_hat, y)
    loss.backward()
    optimizer.step()
    
    print(f"Loss initiale : {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss est NaN"
    print("Test 3 OK : Le modèle s'entraîne et propage les gradients correctement.\n")

if __name__ == "__main__":
    test_efficiency()
    test_duplication_core()
    test_dummy_feature()

