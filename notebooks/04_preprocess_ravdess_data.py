import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm


SR = 16000  # target sampling rate
N_MFCC = 13  # number of MFCC coefficients

def extract_mfcc_mean(filepath, sr=SR, n_mfcc=N_MFCC):
    """
    Loads audio, extracts MFCCs, and returns the time-averaged values.
    """
    try:
        y, _ = librosa.load(filepath, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc, axis=1)
    except Exception as e:
        print(f"Error while loading {filepath}: {e}")
        return None


df_ravdess = pd.read_csv("../data/ravdess_metadata.csv")

X_ravdess = []
y_valence = []
y_arousal = []

for idx, row in tqdm(df_ravdess.iterrows(), total=len(df_ravdess)):
    feats = extract_mfcc_mean(row['file_path'])
    if feats is not None:
        X_ravdess.append(feats)
        y_valence.append(row['valence'])
        y_arousal.append(row['arousal'])

X_ravdess = np.array(X_ravdess)
y_ravdess_valence = np.array(y_valence)
y_ravdess_arousal = np.array(y_arousal)

print(f"X_ravdess shape: {X_ravdess.shape}")
print(f"y_valence shape: {y_ravdess_valence.shape}, distribution: {np.bincount(y_ravdess_valence)}")
print(f"y_arousal shape: {y_ravdess_arousal.shape}, distribution: {np.bincount(y_ravdess_arousal)}")

# Сохраняем
np.save("../data/processed/ravdess_features_mfcc_mean.npy", X_ravdess)
np.save("../data/processed/ravdess_labels_valence.npy", y_ravdess_valence)
np.save("../data/processed/ravdess_labels_arousal.npy", y_ravdess_arousal)