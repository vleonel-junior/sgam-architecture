import json
import pandas as pd
from pathlib import Path

def main():
    """
    Génère un tableau comparatif du nombre de paramètres et du temps de tuning
    entre FT-Transformer et Sparse-FT-Transformer+ pour chaque jeu de données.
    """
    root_dir = Path("output")
    models_to_compare = ["ft_transformer", "sparse_ftt_plus"]
    
    results = []

    # Parcourir tous les jeux de données disponibles dans le répertoire 'output'
    for dataset_dir in sorted(root_dir.iterdir()):
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        dataset_results = {"Dataset": dataset_name}

        # Extraire les informations pour chaque modèle à comparer
        for model_name in models_to_compare:
            stats_path = dataset_dir / model_name / "tuning" / "0" / "stats.json"
            
            if stats_path.exists():
                with stats_path.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extraire le nombre de paramètres et le temps de tuning
                n_params = data.get("best_stats", {}).get("n_parameters")
                total_time = data.get("time")
                
                # Formater pour l'affichage
                dataset_results[f"{model_name} (Params)"] = f"{n_params:,}" if n_params is not None else "N/A"
                dataset_results[f"{model_name} (Tuning Time)"] = total_time if total_time is not None else "N/A"
            else:
                dataset_results[f"{model_name} (Params)"] = "N/A"
                dataset_results[f"{model_name} (Tuning Time)"] = "N/A"
        
        results.append(dataset_results)

    if not results:
        print("Aucun résultat trouvé. Vérifiez la structure de votre répertoire 'output'.")
        return

    # Créer et afficher le DataFrame
    df = pd.DataFrame(results)
    df = df.set_index("Dataset")
    
    print("Tableau Comparatif de Complexité et Temps de Tuning :\n")
    print(df.to_markdown(tablefmt="grid"))

if __name__ == "__main__":
    main()