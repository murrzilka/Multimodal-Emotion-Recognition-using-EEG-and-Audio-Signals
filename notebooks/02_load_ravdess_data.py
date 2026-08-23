import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAVDESS_PATH = "../data/ravdess/audio_speech_actors_01-24"  # path to dir with .wav data

# Emotions mapping RAVDESS -> Binary Labels
# Dictionary: number code of emotion -> (valence, arousal) in binary form
EMOTION_MAP = {
    1: ('neutral', 1, 0),  # valence=1 (positive), arousal=0 (low)
    2: ('calm', 1, 0),  # positive, low arousal
    3: ('happy', 1, 1),  # positive, high arousal
    4: ('sad', 0, 0),  # negative, low arousal
    5: ('angry', 0, 1),  # negative, high arousal
    6: ('fearful', 0, 1),  # negative, high arousal
    7: ('disgust', 0, 1),  # negative, high arousal
    8: ('surprised', 1, 1)  # positive, high arousal
}


def load_ravdess_metadata(root_dir):
    """
    Iterates through the RAVDESS folders, extracts information from the filenames,
    and returns a DataFrame with the following columns:
    - file_path: absolute path to file
    - emotion_code: number from 1 to 8
    - emotion_name: name of the emotion
    - intensity: 1 (normal) or 2 (strong)
    - valence: 0 (negative) or 1 (positive)
    - arousal: 0 (low) or 1 (high)
    - actor: number of actor
    """
    data = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if not file.endswith('.wav'):
                continue

            # Splitting the names by '-'
            parts = file.split('-')

            modality = int(parts[0])
            # only speech (modality = 3)
            if modality != 3:
                continue

            emotion_code = int(parts[2])
            intensity = int(parts[3])
            actor = int(parts[6].split('.')[0])

            if emotion_code not in EMOTION_MAP:
                continue

            emotion_name, valence, arousal = EMOTION_MAP[emotion_code]

            full_path = os.path.join(root, file)
            data.append({
                'file_path': full_path,
                'emotion_code': emotion_code,
                'emotion_name': emotion_name,
                'intensity': intensity,
                'valence': valence,
                'arousal': arousal,
                'actor': actor
            })

    df = pd.DataFrame(data)
    return df


df_ravdess = load_ravdess_metadata(RAVDESS_PATH)
print(f"Loaded {len(df_ravdess)} audiofiles.")
print(df_ravdess.head())
print("\nEmotion Distribution:")
print(df_ravdess['emotion_name'].value_counts())

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.countplot(data=df_ravdess, x='emotion_name', hue='valence')
plt.title('Emotion Valence(0=negative, 1=positive)')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.countplot(data=df_ravdess, x='emotion_name', hue='arousal')
plt.title('Emotion Arousal (0=low, 1=high)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()