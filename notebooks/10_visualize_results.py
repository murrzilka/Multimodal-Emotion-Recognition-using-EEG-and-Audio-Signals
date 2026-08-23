import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

with open("../results/metrics.json", "r", encoding='utf-8') as f:
    all_metrics = json.load(f)

names = list(all_metrics.keys())
accs = [all_metrics[n]['acc'] for n in names]
stds = [all_metrics[n]['std_acc'] for n in names]
f1s = [all_metrics[n]['f1'] for n in names]
f1_stds = [all_metrics[n]['std_f1'] for n in names]

#
# Models compairing
#
x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, accs, width, yerr=stds, label='Accuracy', capsize=5, color='skyblue', edgecolor='black')
bars2 = ax.bar(x + width/2, f1s, width, yerr=f1_stds, label='F1-score', capsize=5, color='lightcoral', edgecolor='black')

ax.set_ylabel('Metric')
ax.set_title('Model comparison (cross-validation, 5 folds)')
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('../results/comparison_bar.png', dpi=300)
plt.show()

# ------------------------
# Confusion matrix for the multimodal model
# ------------------------
if 'Multimodal' in all_metrics:
    y_true = np.array(all_metrics['Multimodal']['y_true'])
    y_pred = np.array(all_metrics['Multimodal']['y_pred'])
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix – Multimodal Model')
    plt.savefig('../results/cm_multimodal.png', dpi=300)
    plt.show()
else:
    print("The 'Multimodal' model was not found in the metrics; the confusion matrix was not generated.")

#
# ROC curves for all models
#
plt.figure(figsize=(8, 6))
for name in names:
    y_true = np.array(all_metrics[name]['y_true'])
    y_prob = np.array(all_metrics[name]['y_prob'])
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curves for all models')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.savefig('../results/roc_curves.png', dpi=300)
plt.show()

print("Visualization is complete. The graphs are saved in ../results/")