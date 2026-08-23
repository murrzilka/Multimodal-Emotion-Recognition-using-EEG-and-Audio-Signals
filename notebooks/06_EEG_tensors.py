import numpy as np
import os
from scipy.stats import zscore


DEAP_PATH = "../data/deap/data_preprocessed_python"
FS_EEG = 128
N_SECONDS_EEG = 4
N_SAMPLES_EEG = FS_EEG * N_SECONDS_EEG
EEG_CHANNELS = 32

def load_eeg_tensors():
    X = []
    y_val = []
    y_aro = []
    for i in range(1, 33):
        filepath = os.path.join(DEAP_PATH, f"s{i:02d}.dat")
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'rb') as f:
            data = np.load(f, allow_pickle=True)
        eeg = data['data'][:, :EEG_CHANNELS, :N_SAMPLES_EEG]
        labels = data['labels']
        for trial in range(40):
            eeg_trial = eeg[trial]
            eeg_trial = zscore(eeg_trial, axis=1)
            X.append(eeg_trial)
            y_val.append(1 if labels[trial, 0] > 5 else 0)
            y_aro.append(1 if labels[trial, 1] > 5 else 0)
    X = np.array(X)
    X = X[:, np.newaxis, :, :]
    return X, np.array(y_val), np.array(y_aro)


X_eeg, y_eeg_val, y_eeg_aro = load_eeg_tensors()
np.save("../data/processed/eeg_tensors.npy", X_eeg)
np.save("../data/processed/eeg_labels_val.npy", y_eeg_val)
np.save("../data/processed/eeg_labels_aro.npy", y_eeg_aro)
print("EEG tensors are saved.", X_eeg.shape)