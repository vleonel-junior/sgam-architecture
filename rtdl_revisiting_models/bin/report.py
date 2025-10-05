import json
from pathlib import Path

import lib

# Configuration
ROOT_DIR = Path("output")
OUTPUT_JSON = "report.json"

def get_task_type_and_metric(dataset_dir: Path):
    """Trouve le type de tâche et la métrique associée pour un dataset."""
    try:
        # On cherche un fichier de configuration pour déduire le type de tâche
        any_config_path = next(dataset_dir.glob("*/tuning/0.toml"))
        config = lib.load_config(any_config_path)
        data_path = lib.get_path(config['base_config']['data']['path'])
        info = lib.load_json(data_path / 'info.json')
        
        task_type = info['task_type']
        metric = 'rmse' if task_type == lib.TaskType.REGRESSION else 'accuracy'
        direction = '↓' if metric == 'rmse' else '↑'
        return f"{direction} {metric}"
    except StopIteration:
        # Fallback si aucun toml n'est trouvé
        return "n/a"

def extract_metric(stats_path: Path):
    """Charge un stats.json et extrait la métrique 'score' du set de test."""
    try:
        with open(stats_path, 'r') as f:
            data = json.load(f)
        # La métrique 'score' est la métrique principale (rmse pour régression, accuracy pour classification)
        return data["metrics"]["test"]["score"]
    except (KeyError, FileNotFoundError, json.JSONDecodeError):
        return None

def main():
    """Script principal pour agréger les résultats."""
    aggregated = {
        "datasets": {},
        "models": {}
    }
    
    # 1. Détecter tous les datasets et leurs métriques
    for dataset_dir in sorted(ROOT_DIR.iterdir()):
        if dataset_dir.is_dir():
            aggregated["datasets"][dataset_dir.name] = get_task_type_and_metric(dataset_dir)

    # 2. Parcourir les résultats pour agréger les scores
    for dataset_name in aggregated["datasets"]:
        dataset_dir = ROOT_DIR / dataset_name
        print(f"Traitement du dataset: {dataset_name}")
        
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            print(f"  - Modèle: {model_name}")
            
            ensemble_dir = model_dir / "tuned_ensemble"
            if not ensemble_dir.exists():
                print(f"    Skip: 'tuned_ensemble' manquant")
                continue
            
            scores = []
            for subdir in ["0_4", "5_9", "10_14"]:
                stats_path = ensemble_dir / subdir / "stats.json"
                score = extract_metric(stats_path)
                if score is not None:
                    scores.append(score)
            
            if scores:
                if model_name not in aggregated["models"]:
                    aggregated["models"][model_name] = {}
                aggregated["models"][model_name][dataset_name] = scores
                print(f"    Scores: {scores}")
            else:
                print(f"    Skip: Aucun score d'ensemble trouvé")

    # 3. Sauvegarder les résultats
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(aggregated, f, indent=4)
    
    print(f"\nJSON généré: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
