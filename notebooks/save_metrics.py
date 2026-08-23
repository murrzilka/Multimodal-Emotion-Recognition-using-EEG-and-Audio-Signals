import json
import os
from datetime import datetime
import numpy as np
def save_metrics_to_json(results_dict, filename="metrics.json"):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            all_data = json.load(f)
    else:
        all_data = {}

    for model_name, metrics in results_dict.items():
        serializable_metrics = {
            'acc': float(metrics['acc']),
            'std_acc': float(metrics['std_acc']),
            'f1': float(metrics['f1']),
            'std_f1': float(metrics['std_f1']),
            'y_true': metrics['y_true'].tolist() if isinstance(metrics['y_true'], np.ndarray) else metrics['y_true'],
            'y_pred': metrics['y_pred'].tolist() if isinstance(metrics['y_pred'], np.ndarray) else metrics['y_pred'],
            'y_prob': metrics['y_prob'].tolist() if isinstance(metrics['y_prob'], np.ndarray) else metrics['y_prob'],
            'timestamp': datetime.now().isoformat()
        }
        all_data[model_name] = serializable_metrics

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print(f"Метрики сохранены в {filename}")