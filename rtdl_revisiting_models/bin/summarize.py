import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind

def main():
    """
    Charge les résultats depuis report.json, effectue une analyse statistique complète
    (moyennes, écarts-types, tests de significativité) et affiche un tableau de
    synthèse au format Markdown, similaire à celui de l'article de recherche.
    """
    report_path = Path("report.json")
    if not report_path.exists():
        print(f"Erreur : Le fichier '{report_path}' est introuvable.")
        print("Veuillez d'abord exécuter le script 'report.py' pour le générer.")
        return

    with report_path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    datasets_info = data["datasets"]
    models_results = data["models"]

    # Mapping des noms de datasets vers leurs abréviations
    dataset_abbreviations = {
        "california_housing": "CA",
        "adult": "AD",
        "helena": "HE",
        "jannis": "JA",
        "higgs_small": "HI",
        "bankchurners": "AB",
        "bankchurners_oversampled": "AB_OS"
    }

    # 1. Calculer les scores moyens et les écarts-types
    stats = {
        model: {
            dataset: {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "scores": scores
            }
            for dataset, scores in results.items()
        }
        for model, results in models_results.items()
    }

    # 2. Créer un DataFrame pour les scores moyens
    mean_scores_df = pd.DataFrame({
        model: {
            dataset: results[dataset]["mean"]
            for dataset in results
        }
        for model, results in stats.items()
    }).T
    mean_scores_df = mean_scores_df.reindex(sorted(mean_scores_df.columns), axis=1)

    # 3. Calculer les classements et déterminer les meilleurs scores (en gras)
    ranks = pd.DataFrame(index=mean_scores_df.index)
    bold_mask = pd.DataFrame(False, index=mean_scores_df.index, columns=mean_scores_df.columns)

    for dataset_name in mean_scores_df.columns:
        metric_direction = datasets_info[dataset_name].split(' ')[0]
        is_rmse = metric_direction == '↓'
        
        # Pour la régression, les scores sont des RMSE négatifs. Un score plus grand (plus proche de 0) est meilleur.
        # La logique doit donc être la même que pour l'accuracy (chercher le max).
        ascending_rank = False  # Toujours classer du plus grand au plus petit score.
        
        # Classement
        ranks[dataset_name] = mean_scores_df[dataset_name].rank(method='min', ascending=ascending_rank)

        # Détermination des meilleurs résultats (en gras)
        best_score_model = mean_scores_df[dataset_name].idxmax()
        bold_mask.loc[best_score_model, dataset_name] = True
        
        best_model_scores = stats[best_score_model][dataset_name]["scores"]

        for model_name in mean_scores_df.index:
            if model_name == best_score_model:
                continue
            
            current_model_scores = stats[model_name][dataset_name]["scores"]
            
            # Test t de Student pour la significativité statistique
            _, p_value = ttest_ind(best_model_scores, current_model_scores, equal_var=False)
            
            if p_value > 0.05:
                bold_mask.loc[model_name, dataset_name] = True

    # 4. Préparer le tableau final pour l'affichage
    display_df = mean_scores_df.copy()
    display_df['rank'] = ranks.mean(axis=1)
    display_df = display_df.sort_values(by='rank')
    
    # Appliquer le formatage (gras et décimales)
    # D'abord, convertir les colonnes de score en type 'object' pour éviter les FutureWarnings
    score_columns = mean_scores_df.columns
    display_df[score_columns] = display_df[score_columns].astype(object)

    for dataset_name in score_columns:
        for model_name in display_df.index:
            # Récupérer la moyenne et l'écart-type depuis le dictionnaire 'stats'
            mean_score = stats[model_name][dataset_name]["mean"]
            std_dev = stats[model_name][dataset_name]["std"]
            
            # Pour la régression, afficher le RMSE positif (valeur absolue)
            display_mean = abs(mean_score) if datasets_info[dataset_name].split(' ')[0] == '↓' else mean_score

            # Créer la chaîne de caractères formatée "moyenne ± écart-type"
            formatted_score = f"{display_mean:.4f} ± {std_dev:.4f}"
            
            if bold_mask.loc[model_name, dataset_name]:
                formatted_score = f"**{formatted_score}**"
            display_df.loc[model_name, dataset_name] = formatted_score

    display_df['rank'] = display_df['rank'].apply(lambda x: f"{x:.1f}")

    # Renommer les colonnes avec les abréviations
    column_headers = {
        name: f"{dataset_abbreviations.get(name, name.upper()[:2])} ({datasets_info[name].split(' ')[0]})"
        for name in mean_scores_df.columns
    }
    column_headers['rank'] = 'rank'
    display_df = display_df.rename(columns=column_headers)

    # 5. Afficher le tableau au format Markdown avec un meilleur formatage pour la lisibilité
    print(f"Tableau d'analyse comparative des modèles (formaté pour Markdown) :\n")
    print(display_df.to_markdown(tablefmt="grid"))

if __name__ == "__main__":
    main()