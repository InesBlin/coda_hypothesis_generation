import subprocess

MODELS = [
    'ada_boost', 'bagging', 'gradient_boosting',
    # 'svc', 
    'mlp_classifier',
    'k_neighbors', 'radius_neighbors', 'ridge', 'sgd',
    'random_forest', 'extra_tree', 
]

for model in MODELS:
    command = f"""
    python experiments/classification/complex/search_hp_complex_classification.py {model} 100 ./data/hypotheses/classification/ ./data/hypotheses/embeds ./experiments/classification/complex/{model}
    """
    subprocess.call(command, shell=True)
    