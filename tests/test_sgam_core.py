import torch
import torch.nn as nn
from sgam.models.sgam_core import SGAM

def test_efficiency_and_categorical():
    """Vérifier que Somme(Contributions) + Baseline == y_hat avec Numériques + Catégorielles"""
    print("--- Test 1 : Axiome d'efficacité avec Num + Cat ---")
    torch.manual_seed(42)
    B, N_num, D = 4, 3, 8
    categories = [5, 10, 2] # 3 variables catégorielles avec 5, 10, 2 modalités
    
    model = SGAM(n_num_features=N_num, categories=categories, d_token=D, d_out=2)
    
    x_num = torch.randn(B, N_num)
    x_cat = torch.tensor([
        [0, 5, 1],
        [4, 9, 0],
        [2, 3, 1],
        [1, 0, 0]
    ], dtype=torch.long)
    
    scores = model.get_importance_scores(x_num=x_num, x_cat=x_cat)
    
    y_hat = scores["y_hat"] # (B, d_out)
    baseline = scores["baseline"] # (d_out,)
    contributions = scores["contributions"] # (B, N_total, d_out)
    
    # Somme des contributions pour chaque sample et chaque classe
    sum_contrib = torch.sum(contributions, dim=1) # (B, d_out)
    
    # Recomposition
    y_hat_recomposed = sum_contrib + baseline.unsqueeze(0)
    
    diff = torch.max(torch.abs(y_hat - y_hat_recomposed)).item()
    print(f"Erreur max de recomposition (Num+Cat) : {diff:.8e}")
    assert diff < 1e-5, f"L'axiome d'efficacité a échoué ! Différence : {diff}"
    print("Test 1 OK : L'attribution additive exacte (Num+Cat) est confirmée.\n")

def test_plr_threshold_recalibration():
    """Vérifier le recalibrage des seuils PLR sur quantiles empiriques"""
    print("--- Test 2 : Recalibrage des seuils PLR ---")
    torch.manual_seed(42)
    N_num, D, N_bins = 2, 4, 4
    model = SGAM(n_num_features=N_num, d_token=D, n_bins=N_bins)
    
    # Données fortement asymétriques
    x_data = torch.empty(100, N_num).exponential_()
    
    # Recalibrage
    model.tokenizer.num_tokenizer.set_thresholds_from_data(x_data)
    
    thresholds = model.tokenizer.num_tokenizer.boundaries
    print(f"Seuils recalibrés (Feature 0) : {thresholds[0].tolist()}")
    assert len(thresholds[0]) == N_bins + 1
    print("Test 2 OK : Recalibrage des quantiles effectué avec succès.\n")

def test_duplication_core():
    """Vérifier que SGAM complet survit à une duplication de feature."""
    print("--- Test 3 : Duplication End-to-End ---")
    torch.manual_seed(42)
    B, N, D = 2, 3, 4
    model = SGAM(n_num_features=N, d_token=D, d_out=1)
    
    # Feature 0 et 1 sont identiques, feature 2 est différente
    x_base = torch.randn(B, 1)
    x_diff = torch.randn(B, 1)
    x_num = torch.cat([x_base, x_base, x_diff], dim=1) # (B, 3)
    
    with torch.no_grad():
        model.tokenizer.num_tokenizer.weight[0] = model.tokenizer.num_tokenizer.weight[1]
        model.tokenizer.num_tokenizer.bias[0] = model.tokenizer.num_tokenizer.bias[1]
        model.local_gate.w_s[0] = model.local_gate.w_s[1]
        model.local_gate.b_s[0] = model.local_gate.b_s[1]
    
    scores = model.get_importance_scores(x_num=x_num)
    importance = scores["importance_l2"]
    
    print(f"Importances: {importance}")
    assert (importance[:, 0] > 1e-4).all(), "Collapse end-to-end (Feature 0)!"
    assert (importance[:, 1] > 1e-4).all(), "Collapse end-to-end (Feature 1)!"
    print("Test 3 OK : La duplication passe sans effondrement ni NaN.\n")

if __name__ == "__main__":
    test_efficiency_and_categorical()
    test_plr_threshold_recalibration()
    test_duplication_core()

