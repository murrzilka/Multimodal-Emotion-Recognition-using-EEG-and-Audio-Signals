**Multimodal Emotion Recognition using EEG and Audio Signals**

**Project Overview**

This project implements a multimodal deep learning system for recognizing human emotions - binary classification: positive/negative valence - by fusing information from two complementary modalities:
    - EEG (electroencephalogram) – brain activity signals from the DEAP dataset
    - Audio (speech) – vocal intonation and prosody from the RAVDESS dataset
The core idea is to demonstrate that combining both modalities yields better accuracy than using either modality alone. Since the two datasets do not contain simultaneous recordings, we artificially create pairs of EEG and audio samples sharing the same emotion label – this serves as a proof‑of‑concept for multimodal fusion.

**Objectives**
    1. Preprocess and extract meaningful features from raw EEG and audio signals.
    2. Train three classification models:
        - EEG-only - using only EEG features
        - Audion-only - using only speech-derived features
        - Multimodal - fusing EEG and audio features via intermediate concatenation or late-fusion transformer
    3. Compare performance using 5-fold stratified cross-validation
    4. Demonstrate that the multimodal approach outperforms each unimodal baseline.

**Datasets**
    1. DEAP (EEG + labels)
        - Description: 32 participants watched 40 one‑minute music videos. EEG (32 channels) and peripheral signals were recorded at 128 Hz.
        - Preprocessing: Already filtered and downsampled. We extract the first 4 seconds (512 samples) from each trial.
        - Labels: Valence ratings from 1 to 9. We binarise at threshold 5 (1 = positive, 0 = negative).
    2. RAVDESS (Audio + labels)
        - Description: 24 professional actors (12 male, 12 female) speaking two sentences with eight emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) at two intensity levels.
        - Preprocessing: We resample to 16 kHz, take the first 3 seconds, and convert to Mel‑spectrograms (128 Mel bands, window 1024, hop 512).
        - Label mapping: Each emotion is mapped to binary valence using a standard mapping (e.g., happy → positive, sad → negative).

**Methodology**
    Data Preprocessing
        - EEG: z-score normalization per channel, only first 4 seconds used
        - Audio: Mel-spectrogram extraction, log-power scaling
    Feature Extraction
        - EEG: Raw time-series tensor of shape (batch, 1, 32, 512) - fed into EEGNet
        - Audio: Mel-spectrogram tensor of shape (batch, 1, 128, time) - fed into a 2D CNN

**Model Architectures**
    1. EEGNet
    A compact convolutional network designed for EEG classification:
        - Temporal convolution (1×64)
        - Spatial convolution (32×1)
        - Depthwise separable convolutions
        - Adaptive average pooling → 32‑dimensional embedding
    2. AudioCNN
    A simple 2D CNN for spectrograms:
        - Three convolutional blocks (32→64→128 filters) with max‑pooling
        - Adaptive average pooling → 128‑dimensional embedding
    3. Multimodal Model - Intermediate Fusion
        - Both EEGNet and AudioCNN are used as feature extractors (their final classification layers are removed).
        - The two embeddings (32 and 128 dims) are concatenated and passed through a fully‑connected classifier with a hidden layer of size 64.
    4. Multimodal – Late‑Fusion Transformer
        - The same EEG and audio embeddings are projected to a common dimension (128) and treated as a sequence of two tokens.
        - A transformer encoder (2 layers, 4 attention heads) models interactions between the two modalities.
        - The first output token is fed into a linear classifier.

**Training Setup**
    1. Loss: Cross‑entropy
    2. Optimizer: Adam (lr = 0.001)
    3. Batch size: 32
    4. Epochs: 50 per fold
    5. Cross‑validation: Stratified 5‑fold to ensure robust evaluation

**Results**

Model	                                    Accuracy (mean ± std)	    F1‑score (mean ± std)
EEG‑only	                                0.8217 ± 0.0135	            0.8225 ± 0.0172
Audio‑only	                                0.7942 ± 0.0317	            0.8073 ± 0.0216
Multimodal(Intermediate Fusion)	            0.8308 ± 0.0155	            0.8313 ± 0.0151
Multimodal(Late‑fusion transformer)	        0.8142 ± 0.0148	            0.8142 ± 0.0082

**Observations:**
    1. EEG‑only achieves a strong baseline (82.2 % accuracy), confirming its strong correlation with emotional valence.
    2. Audio‑only performs well (79.4 % accuracy), better than in earlier experiments, likely due to the chosen spectrogram features and training configuration.
    3. Intermediate Fusion yields the best overall performance (83.1 % accuracy and F1), demonstrating that simple concatenation of modality‑specific embeddings effectively combines complementary information.
    4. Late‑fusion Transformer does not surpass the intermediate fusion approach (81.4 % accuracy). This may be due to the limited number of tokens (only two) and the artificial pairing of samples, which reduces the benefit of complex attention mechanisms. With real paired data, a transformer could potentially exploit richer cross‑modal interactions.

Conclusion: Intermediate fusion (concatenation) is a simple yet powerful strategy for this proof‑of‑concept, providing a clear improvement over unimodal models. The transformer, while conceptually appealing, requires more data or better‑aligned modalities to unlock its full potential.

**Visualisations**
    - Bar chart comparing accuracy and F1 with error bars (results/comparison_bar.png).
    - Confusion matrix for the best model (Intermediate Fusion) (results/cm_multimodal.png).
    - ROC curves for all models (results/roc_curves.png).

**Installation & Usage**
    Requirements
        - Python 3.9+
        - PyTorch 1.10+
        - NumPy, SciPy, Pandas
        - Librosa, MNE
        - Scikit‑learn, Matplotlib, Seaborn
    Install dependencies:
        pip install -r requirements.txt

**Future Work**
    - Real paired data (crucial) - collect simultaneous EEG and speech recordings for fully aligned multimodal learning.
    - Advanced fusion - explore cross‑modal attention, multimodal transformers, and contrastive learning.
    - Multi‑task learning - predict both valence and arousal simultaneously.
    - Data augmentation - apply noise injection, time‑shifting, and spectrogram masking to improve generalisation.

**Conclusion**
We have successfully demonstrated that multimodal emotion recognition combining EEG and audio signals outperforms unimodal classifiers. Among the fusion strategies tested, intermediate concatenation proved most effective on the artificial paired data, achieving an accuracy of 83.1 %. The late‑fusion transformer, while promising, did not surpass the simpler approach, likely due to the limited amount of modality interactions in the artificially constructed pairs. This work provides a solid proof‑of‑concept and a baseline for future research on real‑world affective brain‑computer interfaces.

**Acknowledgements**
    - DEAP dataset: Koelstra et al., 2012.
    - RAVDESS dataset: Livingstone & Russo, 2018.
    - PyTorch and open‑source libraries.