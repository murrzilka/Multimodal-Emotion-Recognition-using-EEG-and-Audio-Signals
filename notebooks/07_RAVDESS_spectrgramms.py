import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm

SR_AUDIO = 16000
N_SECONDS_AUDIO = 3
N_SAMPLES_AUDIO = SR_AUDIO * N_SECONDS_AUDIO  # 48000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512

def extract_melspectrogram(filepath, sr=SR_AUDIO, duration=N_SECONDS_AUDIO):
    y, _ = librosa.load(filepath, sr=sr, duration=duration)
    # Если аудио короче, дополняем нулями
    # If audion is shorter, padding with zeros
    if len(y) < N_SAMPLES_AUDIO:
        y = np.pad(y, (0, N_SAMPLES_AUDIO - len(y)))
    else:
        y = y[:N_SAMPLES_AUDIO]
    # Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                         n_fft=N_FFT, hop_length=HOP_LENGTH)
    # Take the logarithm
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db  # (128, time)

df_rav = pd.read_csv("../data/ravdess_metadata.csv")

X_audio = []
y_audio_val = []
y_audio_aro = []

for idx, row in tqdm(df_rav.iterrows(), total=len(df_rav)):
    mel = extract_melspectrogram(row['file_path'])
    if mel is not None:
        X_audio.append(mel)
        y_audio_val.append(row['valence'])
        y_audio_aro.append(row['arousal'])

X_audio = np.array(X_audio)  # (n, 128, time)
X_audio = X_audio[:, np.newaxis, :, :]  # (n, 1, 128, time)
y_audio_val = np.array(y_audio_val)
y_audio_aro = np.array(y_audio_aro)

np.save("../data/processed/audio_spectrograms.npy", X_audio)
np.save("../data/processed/audio_labels_val.npy", y_audio_val)
np.save("../data/processed/audio_labels_aro.npy", y_audio_aro)
print("Audio spectrograms have been saved..", X_audio.shape)