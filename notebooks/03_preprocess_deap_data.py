import numpy as np
import scipy.signal as signal
from scipy.stats import zscore
import os

DEAP_PATH = "../data/deap/data_preprocessed_python"
FS = 128  # sampling rate in DEAP
EEG_CHANNELS = 32  # using 32 channel from DEAP

BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}


def bandpower(data, fs, band):
    """
    Calculates the average signal power within a specified frequency band.
    data: 1D array of signal
    fs: sample rate
    band: tuple (low, high)
    """
    low, high = band
    freqs, psd = signal.welch(data, fs, nperseg=fs * 2)  # 2 seconds window
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.mean(psd[idx])  # average power in the band


def preprocess_deap_participant(filepath):
    """
    Loads data for a single participant, filters it, normalizes it,
    and extracts spectral features for each trial.
    Returns: features (40 trials, n_features) and labels (40, 2) [valence, arousal]
    """
    with open(filepath, 'rb') as f:
        data = np.load(f, allow_pickle=True)
    eeg = data['data']
    labels = data['labels']

    # Using onlu EEG channels
    eeg = eeg[:, :EEG_CHANNELS, :]

    features = []
    for trial in range(eeg.shape[0]):
        trial_features = []
        for ch in range(eeg.shape[1]):
            sig = eeg[trial, ch, :]
            # z-score normaliztion for each channel
            sig = zscore(sig)
            for band_name, band_range in BANDS.items():
                power = bandpower(sig, FS, band_range)
                trial_features.append(power)
        features.append(trial_features)

    features = np.array(features)
    # labels: valence и arousal, binarization with a threshold of 5
    valence = (labels[:, 0] > 5).astype(int)  # 1 if >5, else 0
    arousal = (labels[:, 1] > 5).astype(int)
    return features, valence, arousal


all_features = []
all_valence = []
all_arousal = []

for i in range(1, 33):
    filepath = os.path.join(DEAP_PATH, f"s{i:02d}.dat")
    if not os.path.exists(filepath):
        print(f"File {filepath} was not found, skipping")
        continue
    print(f"Participant processing {i}")
    feats, val, aro = preprocess_deap_participant(filepath)
    all_features.append(feats)
    all_valence.append(val)
    all_arousal.append(aro)

X_deap = np.vstack(all_features)
y_deap_valence = np.hstack(all_valence)
y_deap_arousal = np.hstack(all_arousal)

print(f"X_deap shape: {X_deap.shape}")
print(f"y_valence shape: {y_deap_valence.shape}, distribution: {np.bincount(y_deap_valence)}")
print(f"y_arousal shape: {y_deap_arousal.shape}, distribution: {np.bincount(y_deap_arousal)}")

np.save("../data/processed/deap_features.npy", X_deap)
np.save("../data/processed/deap_labels_valence.npy", y_deap_valence)
np.save("../data/processed/deap_labels_arousal.npy", y_deap_arousal)