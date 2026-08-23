import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import joblib
import json

#
# Data loading
#
# DEAP (EEG)
X_deap = np.load("../data/processed/deap_features.npy")
y_deap_val = np.load("../data/processed/deap_labels_valence.npy")
y_deap_ar = np.load("../data/processed/deap_labels_arousal.npy")

# RAVDESS (audio)
X_rav = np.load("../data/processed/ravdess_features_mfcc_mean.npy")
y_rav_val = np.load("../data/processed/ravdess_labels_valence.npy")
y_rav_ar = np.load("../data/processed/ravdess_labels_arousal.npy")

# Valency classification
# Picking labels for training
y_deap = y_deap_val
y_rav = y_rav_val

print("Sample sizes:")
print(f"DEAP: {X_deap.shape[0]} samples, classes: {np.bincount(y_deap)}")
print(f"RAVDESS: {X_rav.shape[0]} samples, classes: {np.bincount(y_rav)}")


#
# Creation of artificial pairs
#
def create_artificial_pairs(X1, y1, X2, y2, n_pairs_per_class=500, random_state=42):
    """
    Creates artificial pairs (X1, X2) with identical labels y.
    For each class, n_pairs_per_class random pairs are selected.
    Returns: X1_pairs, X2_pairs, y_pairs
    """
    np.random.seed(random_state)
    classes = np.unique(y1)
    X1_pairs, X2_pairs, y_pairs = [], [], []

    for cls in classes:
        idx1 = np.where(y1 == cls)[0]
        idx2 = np.where(y2 == cls)[0]
        if len(idx1) < n_pairs_per_class:
            idx1 = np.random.choice(idx1, n_pairs_per_class, replace=True)
        else:
            idx1 = np.random.choice(idx1, n_pairs_per_class, replace=False)
        if len(idx2) < n_pairs_per_class:
            idx2 = np.random.choice(idx2, n_pairs_per_class, replace=True)
        else:
            idx2 = np.random.choice(idx2, n_pairs_per_class, replace=False)

        for i in range(n_pairs_per_class):
            X1_pairs.append(X1[idx1[i]])
            X2_pairs.append(X2[idx2[i]])
            y_pairs.append(cls)

    X1_pairs = np.array(X1_pairs)
    X2_pairs = np.array(X2_pairs)
    y_pairs = np.array(y_pairs)

    perm = np.random.permutation(len(y_pairs))
    return X1_pairs[perm], X2_pairs[perm], y_pairs[perm]


X_eeg, X_audio, y = create_artificial_pairs(X_deap, y_deap, X_rav, y_rav, n_pairs_per_class=500)
print(f"Created {len(y)} artificial pairs.")
print(f"Class distribution: {np.bincount(y)}")


X_eeg_train, X_eeg_test, X_audio_train, X_audio_test, y_train, y_test = train_test_split(
    X_eeg, X_audio, y, test_size=0.2, random_state=42, stratify=y
)

scaler_eeg = StandardScaler()
scaler_audio = StandardScaler()

X_eeg_train_scaled = scaler_eeg.fit_transform(X_eeg_train)
X_eeg_test_scaled = scaler_eeg.transform(X_eeg_test)

X_audio_train_scaled = scaler_audio.fit_transform(X_audio_train)
X_audio_test_scaled = scaler_audio.transform(X_audio_test)

# Multimodal features - concatenation
X_multi_train = np.concatenate([X_eeg_train_scaled, X_audio_train_scaled], axis=1)
X_multi_test = np.concatenate([X_eeg_test_scaled, X_audio_test_scaled], axis=1)

#
# Model training (using MLPClassifier for non-linearity)
#
models = {
    'EEG-only': X_eeg_train_scaled,
    'Audio-only': X_audio_train_scaled,
    'Multimodal': X_multi_train
}
test_sets = {
    'EEG-only': X_eeg_test_scaled,
    'Audio-only': X_audio_test_scaled,
    'Multimodal': X_multi_test
}

results = {}

for name, X_train in models.items():
    print(f"\n--- Training {name} ---")
    clf = MLPClassifier(hidden_layer_sizes=(64,), activation='relu',
                        max_iter=500, random_state=42, early_stopping=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(test_sets[name])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary')
    results[name] = {'accuracy': acc, 'f1': f1, 'model': clf}
    print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

#
# Comparison of results
#
print("\n===== Comparison of models =====")
for name, metrics in results.items():
    print(f"{name:15} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")


# Accuracy comparison chart
plt.figure(figsize=(8, 5))
names = list(results.keys())
accuracies = [results[n]['accuracy'] for n in names]
f1_scores = [results[n]['f1'] for n in names]

x = np.arange(len(names))
width = 0.35
plt.bar(x - width / 2, accuracies, width, label='Accuracy')
plt.bar(x + width / 2, f1_scores, width, label='F1-score')
plt.xlabel('Model')
plt.ylabel('Metrics')
plt.title('Comparison of accuracy and F1-score on synthetic pairs')
plt.xticks(x, names)
plt.legend()
plt.tight_layout()
plt.savefig('../results/comparison_MLP.png')
plt.show()

metrics_to_save = {name: {k: v for k, v in metrics.items() if k != 'model'}
                   for name, metrics in results.items()}

with open('../results/metrics_MLP.json', 'w') as f:
    json.dump(metrics_to_save, f, indent=4)

for name, metrics in results.items():
    joblib.dump(metrics['model'], f'../results/model_{name}.pkl')

print("Metrics saved in results/metrics_MLP.json")
print("Models saved as results/model_*.pkl")