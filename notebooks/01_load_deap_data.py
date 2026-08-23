import numpy as np
import matplotlib.pyplot as plt

def load_deap_data(filepath):
    """
    Loading data of one participant from DEAP.
    filepath: path to file sXX.dat
    """
    with open(filepath, 'rb') as f:
        data = np.load(f, allow_pickle=True)
    eeg = data['data']
    labels = data['labels']
    return eeg, labels

# Loading of first participant from dataset
eeg, labels = load_deap_data('../data/data_preprocessed_python/s01.dat')
print(f"EEG data shape: {eeg.shape}")
print(f"Labels shape: {labels.shape}")

valence = labels[:, 0]  # valence
arousal = labels[:, 1]  # arousal

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.hist(valence, bins=9, color='skyblue', edgecolor='black')
plt.title('Valence Distribution (1-9)')
plt.xlabel('Valence')
plt.subplot(1, 2, 2)
plt.hist(arousal, bins=9, color='salmon', edgecolor='black')
plt.title('Arousal Distribution (1-9)')
plt.xlabel('Arousal')
plt.show()

# EEG graph for one channel and one trial
channel = 0
trial = 0
signal = eeg[trial, channel, :]  # 8064 points

plt.figure(figsize=(14, 4))
plt.plot(signal)
plt.title(f'EEG Signal - Trial {trial}, Channel {channel}')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.show()