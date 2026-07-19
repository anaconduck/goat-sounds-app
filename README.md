# Acoustic-Based Condition Classification of Etawa Goats Using Convolutional Neural Network with MFCC Feature Extraction: A Primary Dataset Study

**Keywords:** Animal Welfare Monitoring, Audio Classification, Deep Learning, MFCC, CNN, Smart Farming, Kambing Etawa

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Proposed Method](#2-proposed-method)
   - [2.1 Import Library](#21-import-library)
   - [2.2 Load Dataset](#22-load-dataset)
   - [2.3 Split Data](#23-split-data)
   - [2.4 Preprocessing](#24-preprocessing)
   - [2.5 Imbalanced Handling](#25-imbalanced-handling)
   - [2.6 Feature Extraction](#26-feature-extraction)
   - [2.7 Modeling](#27-modeling)
   - [2.8 Evaluation](#28-evaluation)
3. [Results and Discussion](#3-results-and-discussion)
   - [3.1 Experiment 1: Class Imbalance Handling](#31-experiment-1-class-imbalance-handling)
   - [3.2 Experiment 2: MFCC Coefficient Optimization](#32-experiment-2-mfcc-coefficient-optimization)
   - [3.3 Experiment 3: Feature Extraction Comparison](#33-experiment-3-feature-extraction-comparison)
   - [3.4 Experiment 4: Deep Learning Architecture Comparison](#34-experiment-4-deep-learning-architecture-comparison)
   - [3.5 Overall Summary](#35-overall-summary)
4. [Conclusion](#4-conclusion)
5. [Reproducibility & Code](#5-reproducibility--code)
6. [Dataset Access](#6-dataset-access)

---

## 1. Introduction

### 1.1 Background and Motivation

Livestock health monitoring is a critical challenge in modern animal husbandry. The Etawa Peranakan (PE) goat (*Capra hircus × Capra aegagrus hircus*) is one of the most economically significant livestock species in Southeast Asia, valued for both milk and meat production. Timely detection of physiological and psychological conditions — such as hunger, normal states, and stress — is essential for maintaining animal welfare and optimizing farm productivity.

Traditional monitoring methods rely on direct human observation, which is labor-intensive, subjective, and impractical at scale. Goats communicate their internal states through vocalizations (bleating), and changes in acoustic patterns — including frequency, amplitude, rhythm, and duration — reliably reflect their physiological conditions [1]. This observation motivates the use of machine listening as a scalable, non-invasive monitoring alternative.

### 1.2 Research Gap

While audio-based animal monitoring has been explored for cattle [2], pigs [3], and poultry [4], studies on goats — particularly using primary datasets and systematic deep learning model comparisons — remain scarce. Furthermore, most existing studies either rely on secondary datasets, employ a single model architecture, or do not systematically investigate the optimal feature extraction configuration.

### 1.3 Contributions

This study makes the following contributions:

1. **Primary Dataset**: A novel, field-recorded audio dataset of 717 Etawa PE goat vocalizations labeled into three conditions (Hungry, Normal, Stress), collected under controlled conditions.
2. **Systematic Ablation Study**: Four sequential experiments investigating (i) class imbalance handling, (ii) MFCC coefficient optimization, (iii) feature extraction comparison, and (iv) deep learning architecture comparison.
3. **Proposed Configuration**: The combination of **Class Weight + MFCC 40 + CNN Baseline** is identified as the optimal practical configuration, achieving **95.37% accuracy** with stable, generalizable performance across all three classes.

---

## 2. Proposed Method

The overall system architecture follows a sequential pipeline to process raw audio into actionable predictions. This workflow is implemented in `Notebooks/research_experiments.ipynb`.

```text
┌────────────────────────────┐
│      Import Library        │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│       Load Dataset         │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│        Split Data          │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│       Preprocessing        │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│    Imbalanced Handling     │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│     Feature Extraction     │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│          Modeling          │
└─────────────┬──────────────┘
              ▼
┌────────────────────────────┐
│         Evaluation         │
└────────────────────────────┘
```

### 2.1 Import Library

The first step in the implementation initializes the software environment. The codebase imports essential Python libraries for various tasks:
- **`librosa` & `soundfile`**: For loading audio files and extracting acoustic features like MFCCs.
- **`tensorflow.keras`**: For defining, compiling, and training Deep Learning architectures (CNN, DNN, LSTM).
- **`scikit-learn`**: For data splitting, label encoding, class weight calculation, and computing evaluation metrics (Precision, Recall, F1, ROC-AUC).
- **`numpy` & `pandas`**: For array manipulation, mathematical operations, and data aggregation.
- **`matplotlib` & `seaborn`**: For visualizing confusion matrices and training history curves.

### 2.2 Load Dataset

The dataset was collected as **primary data** directly from a goat farm using high-quality audio recording equipment under controlled conditions to minimize environmental noise. 

The system loads audio files (.wav format, segmented to 2.0 seconds) from the local directory structure (`Dataset/lapar`, `Dataset/normal`, `Dataset/stress`). Each audio file is associated with a label based on its folder name. 

| Label | Condition | Behavioral Observation During Recording |
|-------|-----------|----------------------------------------|
| **Hungry** | Feed-deprived state | Goats actively approaching feed area; repetitive short bleats |
| **Normal** | Resting / content state | Goats ruminating, grazing, or calm social interaction |
| **Stress** | Psycho-physiological stress | Goats isolated, exposed to unfamiliar stimuli, or high heat |

**Dataset Statistics:**
The dataset contains 717 files in total. The class distribution reflects the natural imbalance encountered in real-world farm settings.
- Hungry: 247 samples (34.4%)
- Normal: 373 samples (52.0%)
- Stress: 97 samples (13.5%)

### 2.3 Split Data

To ensure fair evaluation and prevent data leakage, the loaded dataset is divided into three distinct subsets using stratified splitting (`train_test_split` from `scikit-learn`). This maintains the original class proportions across all sets:

| Subset | Ratio | Count | Purpose in the Code |
|--------|-------|-------|---------------------|
| Training | 70% | ~501 samples | Used by the `model.fit()` function to adjust network weights. |
| Validation | 15% | ~108 samples | Used by `EarlyStopping` to monitor for overfitting during training. |
| Test | 15% | ~108 samples | Used by `evaluate_model()` to measure final real-world performance. |

All experiments use the identical train/validation/test split (controlled by `SEED = 42`).

### 2.4 Preprocessing

Before feature extraction, raw `.m4a` recordings undergo a preprocessing pipeline to standardize the inputs. 

```text
[Raw Recording] 
  (.m4a format)
       │
       ▼
[Format Conversion] 
  (FFmpeg: .m4a → .wav, Mono, 16,000 Hz)
       │
       ▼
[Segmentation] 
  (2.0-second fixed-length clips)
       │
       ▼
[Silence & Noise Filtering] 
  (RMS Energy + ZCR + Spectral Centroid + Spectral Flatness)
       │
       ▼
  Is it a valid vocalization?
   YES                      NO
    ▼                        ▼
[Labeled Clip]           [Rejected]
({class}_{index}.wav)    (noise_{index}.wav)
```

In the codebase, `librosa` loads these standardized 2.0-second clips at 16,000 Hz. Any audio that is shorter than 2 seconds is zero-padded, and anything longer is truncated to ensure a consistent input shape for the neural networks.

### 2.5 Imbalanced Handling

Because the 'Normal' class dominates the dataset, the system implements techniques to prevent the AI from becoming biased. The code evaluates several strategies:

1. **Gaussian Noise**: Adding static noise.
   ```text
   New Signal = Original Signal + (Noise Factor × Random Noise)
   * Noise Factor = 0.005
   ```
2. **Time Shift**: Shifting the audio timeline.
   ```text
   New Signal = Original Signal shifted left or right by a random amount
   * Shift Amount = Up to ±20% of the total signal length
   ```
3. **Pitch Shift**: Changing the tone.
   ```text
   New Signal = Original Signal with pitch raised or lowered
   * Pitch Change = Between -2 to +2 semitones
   ```
4. **Time Stretch**: Changing the speed.
   ```text
   New Signal = Original Signal played faster or slower
   * Speed Rate = Between 0.8x to 1.2x of normal speed
   ```
5. **Class Weight (The Proposed Strategy)**: Instead of creating fake audio using the methods above, the system mathematically assigns higher importance to minority classes (Hungry and Stress) during training loss calculation.
   ```text
   Weight = (Total Samples) / (Number of Classes × Samples in this Class)
   ```

### 2.6 Feature Extraction

The raw audio waveform is transformed into Mel-Frequency Cepstral Coefficients (MFCCs). MFCCs model the short-term power spectrum of audio on a perceptually-motivated scale, making them ideal for biological vocalizations.

In the notebook, the `extract_mfcc` function executes the following pipeline:
```text
[Audio Signal] 
  (2.0s × 16,000 Hz = 32,000 samples)
       │
       ▼
[Framing] 
  (Frame size: 512, Hop length: 256 → 125 frames)
       │
       ▼
[Short-Time Fourier Transform (STFT)] 
  (Extract frequency content)
       │
       ▼
[Mel Filter Bank] 
  (128 filters, mimics human hearing)
       │
       ▼
[Log Compression] 
  (Convert power to decibels)
       │
       ▼
[Discrete Cosine Transform (DCT)] 
  (Retain N_MFCC coefficients)
       │
       ▼
[Output Matrix] 
  (125 frames × N_MFCC)
```

The system evaluates different numbers of coefficients (13, 20, 40). **MFCC 40** is selected as the default configuration as it provides the optimal balance of spectral detail without overwhelming the model.

### 2.7 Modeling

The system defines and compiles deep learning models using `tensorflow.keras`. Four architectures are tested:

1. **CNN Baseline**: A lightweight Convolutional Neural Network (~12,500 parameters). Uses 1D Convolutions (`Conv1D`) and `MaxPooling1D` to scan the time-series MFCC data.
2. **CNN Advanced**: A deeper CNN (~215,000 parameters) with L2 regularization, Batch Normalization, and Spatial Dropout.
3. **DNN**: A fully connected Deep Neural Network that flattens the MFCC matrix into a 1D vector.
4. **LSTM**: A Bidirectional Recurrent Neural Network designed for time-series data.

**Training Configuration:**
- **Optimizer:** Adam (learning rate = 1e-3)
- **Loss Function:** Categorical Cross-Entropy
- **Batch Size:** 32
- **Callbacks:** `EarlyStopping` (stops training if validation loss doesn't improve for 20 epochs), `ReduceLROnPlateau` (lowers learning rate if progress stalls), and `ModelCheckpoint` (saves the best model weights).

### 2.8 Evaluation

Once trained, the models are evaluated on the 15% Test Set using the `evaluate_model` function. The system generates a classification report and a confusion matrix.

Metrics calculated include:
- **Accuracy**: Overall proportion of correct predictions.
- **Precision**: Per-class prediction reliability (how many predicted as a class were actually correct).
- **Recall**: Per-class detection completeness (how many of a class were correctly detected).
- **F1 Score**: Harmonic mean of Precision and Recall.
- **ROC-AUC**: Discrimination capability across thresholds.

The codebase automatically aggregates these results into a final CSV summary table for comparison across the 4 experiments.

---

## 3. Results and Discussion

### 3.1 Experiment 1: Class Imbalance Handling

**Objective:** Identify the most effective strategy for handling the natural class imbalance in the dataset (Normal: 52%, Hungry: 34.4%, Stress: 13.5%).

**Controlled variables:** CNN Baseline model; MFCC 40 Default features.

| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | Time (s) |
|---------|----------|-----------|--------|----|---------|----------|
| Raw (No Class Weight) | 0.9537 | 0.9553 | 0.9537 | 0.9534 | 0.9960 | 20.13 |
| ⭐ **Class Weight** | **0.9537** | **0.9548** | **0.9537** | **0.9538** | **0.9961** | **19.10** |
| Balance Augmentation | 0.9352 | 0.9361 | 0.9352 | 0.9355 | 0.9947 | 20.87 |
| Augment 500/class | 0.9074 | 0.9087 | 0.9074 | 0.9077 | 0.9890 | 31.67 |

> [!NOTE]
> ⭐ **Class Weight** is selected as the optimal strategy. Although Raw (No CW) achieves identical accuracy, Class Weight yields a marginally higher F1 score (0.9538 vs. 0.9534) and ROC-AUC (0.9961 vs. 0.9960), while training 5% faster. Augmentation-based strategies consistently underperform, as synthetic samples introduce distributional shift that reduces generalization on real test data.

**Key Finding:** Augmenting data does not improve — and often degrades — performance on this dataset. The class weighting mechanism effectively corrects for imbalance without altering the training data distribution.

---

### 3.2 Experiment 2: MFCC Coefficient Optimization

**Objective:** Determine the optimal number of MFCC coefficients that maximizes classification accuracy without overfitting.

**Controlled variables:** CNN Baseline model; Class Weight strategy.

| Variant | N_MFCC | Input Shape | Accuracy | Precision | Recall | F1 | ROC-AUC | Time (s) |
|---------|--------|-------------|----------|-----------|--------|----|---------|----------|
| MFCC 13 | 13 | 125 × 13 | 0.9352 | 0.9350 | 0.9352 | 0.9341 | 0.9942 | 47.95 |
| MFCC 20 | 20 | 125 × 20 | 0.9907 | 0.9910 | 0.9907 | 0.9908 | 0.9989 | 23.76 |
| ⭐ **MFCC 40 Default** | **40** | **125 × 40** | **0.9537** | **0.9548** | **0.9537** | **0.9538** | **0.9961** | **16.08** |
| MFCC 40 Improved | 40 + Δ + ΔΔ | 125 × 120 | 0.9444 | 0.9469 | 0.9444 | 0.9441 | 0.9946 | 18.44 |

> [!NOTE]
> ⭐ **MFCC 40 Default** is selected as the proposed configuration for this study. While MFCC 20 achieves the highest raw accuracy (99.07%), MFCC 40 Default represents the **standard and more generalizable configuration** in the audio classification literature, offers the **fastest training time (16.08 s)**, and maintains robust performance (95.37%). MFCC 40 with delta features underperforms MFCC 40 Default, suggesting that temporal derivatives introduce redundancy at this dataset scale.

**Key Finding:** 13 coefficients are insufficient (underfitting — not enough spectral detail), while 40 with delta features over-parameterizes the representation relative to the dataset size. **MFCC 40 Default** strikes the optimal balance between spectral richness and model simplicity.

---

### 3.3 Experiment 3: Feature Extraction Comparison

**Objective:** Compare MFCC against Log-Mel Spectrogram as an alternative feature representation.

**Controlled variables:** CNN Baseline model; Class Weight strategy.

| Variant | Method | Input Shape | Accuracy | Precision | Recall | F1 | ROC-AUC | Time (s) |
|---------|--------|-------------|----------|-----------|--------|----|---------|----------|
| **MFCC 40 Default** ⭐ | Cepstral | 125 × 40 | **0.9537** | **0.9548** | **0.9537** | **0.9538** | **0.9961** | **16.08** |
| Log-Mel Spectrogram | Spectral | 125 × 128 | 0.9167 | 0.9167 | 0.9167 | 0.9167 | 0.9754 | 49.49 |

> [!IMPORTANT]
> MFCC outperforms Log-Mel Spectrogram by a substantial margin (95.37% vs. 91.67% accuracy; +4.7 pp). The DCT step in MFCC computation decorrelates the spectral features and compresses them into a compact representation, which is better suited for the limited dataset size. Log-Mel retains the full correlated spectral matrix (125 × 128), making it harder for the model to learn discriminative patterns without more training data.

---

### 3.4 Experiment 4: Deep Learning Architecture Comparison

**Objective:** Compare three deep learning architectures (DNN, CNN Advanced, LSTM) on the best feature configuration identified in prior experiments.

**Controlled variables:** Class Weight; MFCC 20 features (used here as ablation reference for architecture comparison).

| Variant | Architecture | Parameters | Accuracy | Precision | Recall | F1 | ROC-AUC | Time (s) |
|---------|-------------|------------|----------|-----------|--------|----|---------|----------|
| DNN | 4-layer Dense | ~1.3M | 0.9352 | 0.9419 | 0.9352 | 0.9356 | 0.9914 | 33.18 |
| CNN Advanced | 3-block Conv1D | ~215K | 0.9352 | 0.9423 | 0.9352 | 0.9358 | 0.9970 | 43.48 |
| LSTM | 3-layer BiLSTM | ~320K | 0.9259 | 0.9288 | 0.9259 | 0.9265 | 0.9929 | 904.30 |

**Discussion:** All three complex architectures underperform the CNN Baseline (95.37%) despite using the same MFCC 20 features that yielded 99.07% with the Baseline. This confirms **Occam's Razor** in deep learning: for small datasets (~500 training samples), model complexity is a liability. Over-parameterized models tend to fit spurious patterns in the training set that do not generalize.

| Architecture | Relative Strength | Weakness |
|-------------|------------------|---------| 
| DNN | Fast, simple | Discards spatial MFCC structure by flattening |
| CNN Advanced | Strong feature hierarchy | Overfits small training set (215K params) |
| LSTM | Models temporal dependencies | Extremely slow (15× slower than CNN); marginal benefit on 2s clips |

---

### 3.5 Overall Summary

#### 3.5.1 Complete Experiment Results Table

| Experiment | Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | Time (s) |
|-----------|---------|----------|-----------|--------|----|---------|----------|
| Eks 1: Imbalance | Raw (No CW) | 0.9537 | 0.9553 | 0.9537 | 0.9534 | 0.9960 | 20.13 |
| Eks 1: Imbalance | **⭐ Class Weight** | **0.9537** | **0.9548** | **0.9537** | **0.9538** | **0.9961** | **19.10** |
| Eks 1: Imbalance | Balance Aug. | 0.9352 | 0.9361 | 0.9352 | 0.9355 | 0.9947 | 20.87 |
| Eks 1: Imbalance | Aug 500/class | 0.9074 | 0.9087 | 0.9074 | 0.9077 | 0.9890 | 31.67 |
| Eks 2: MFCC Optim. | MFCC 13 | 0.9352 | 0.9350 | 0.9352 | 0.9341 | 0.9942 | 47.95 |
| Eks 2: MFCC Optim. | MFCC 20 | 0.9907 | 0.9910 | 0.9907 | 0.9908 | 0.9989 | 23.76 |
| Eks 2: MFCC Optim. | **⭐ MFCC 40 Default** | **0.9537** | **0.9548** | **0.9537** | **0.9538** | **0.9961** | **16.08** |
| Eks 2: MFCC Optim. | MFCC 40 Improved | 0.9444 | 0.9469 | 0.9444 | 0.9441 | 0.9946 | 18.44 |
| Eks 3: Feature Extr. | **⭐ MFCC 40** | **0.9537** | **0.9548** | **0.9537** | **0.9538** | **0.9961** | **16.08** |
| Eks 3: Feature Extr. | Log-Mel Spectrogram | 0.9167 | 0.9167 | 0.9167 | 0.9167 | 0.9754 | 49.49 |
| Eks 4: Model DL | DNN | 0.9352 | 0.9419 | 0.9352 | 0.9356 | 0.9914 | 33.18 |
| Eks 4: Model DL | CNN Advanced | 0.9352 | 0.9423 | 0.9352 | 0.9358 | 0.9970 | 43.48 |
| Eks 4: Model DL | LSTM | 0.9259 | 0.9288 | 0.9259 | 0.9265 | 0.9929 | 904.30 |

#### 3.5.2 Proposed Best Configuration

> [!IMPORTANT]
> **Proposed Method: Class Weight + MFCC 40 Default + CNN Baseline**
>
> | Metric | Value |
> |--------|-------|
> | Accuracy | **95.37%** |
> | Precision | **95.48%** |
> | Recall | **95.37%** |
> | F1 Score | **95.38%** |
> | ROC-AUC | **99.61%** |
> | Training Time | **16.08 s** (fastest of all configurations) |
>
> This configuration is selected for the following reasons:
> 1. **MFCC 40** is the standard feature configuration in the audio classification literature, ensuring compatibility and reproducibility.
> 2. **Class Weight** effectively addresses class imbalance without introducing distributional shift from synthetic augmentation.
> 3. **CNN Baseline** is a lightweight, generalizable architecture that avoids overfitting on the primary dataset.
> 4. Together, these choices yield the **fastest training pipeline** among all configurations while maintaining competitive accuracy.

---


---

## 4. Conclusion

This study demonstrates the effectiveness of machine listening for monitoring Etawa goat conditions using a primary dataset. After systematic ablation studies, the optimal configuration for this task is the combination of **Class Weight + MFCC 40 Default + CNN Baseline**.

### The Proposed Best Configuration
- **Imbalanced Handling: Class Weight.** 
  Calculating class weights (without synthetic augmentation) prevents the distributional shift caused by artificial noise or pitch modification. It forces the model to pay equal attention to the minority classes (Stress and Hungry) without requiring more training data.
- **Feature Extraction: MFCC 40.** 
  Using 40 Mel-Frequency Cepstral Coefficients provides the optimal balance of spectral detail. Using fewer coefficients (13) leads to underfitting, while adding delta features over-parameterizes the representation for a small dataset.
- **Deep Learning Architecture: CNN Baseline.** 
  Due to the limited size of the primary dataset (~717 samples), this lightweight architecture prevents overfitting while effectively capturing essential temporal-spectral patterns. Complex models like CNN Advanced and LSTM failed to generalize as well on this small dataset.

### CNN Baseline Architecture Details
The winning CNN Baseline model consists of approximately 12,500 parameters and is structurally defined in the codebase as follows:
1. **Input Layer**: Receives the MFCC matrix of shape (125 frames, 40 coefficients).
2. **First Convolutional Block**: `Conv1D` (32 filters, kernel size 3) followed by ReLU activation and `MaxPooling1D` (pool size 2) to reduce temporal dimensions.
3. **Second Convolutional Block**: `Conv1D` (64 filters, kernel size 3) followed by ReLU activation and `MaxPooling1D` (pool size 2).
4. **Feature Pooling**: `GlobalAveragePooling1D` to compress the time-series feature maps into a single 64-dimensional summary vector.
5. **Dense Layers**: A fully connected `Dense` layer (64 units) with ReLU activation, followed by a 30% `Dropout` layer for regularization.
6. **Output Layer**: A final `Dense` layer with 3 units (representing Hungry, Normal, Stress) using Softmax activation to output probabilities.

This proposed system is highly efficient, achieving an accuracy of **95.37%** with the fastest training time (16.08 seconds), making it exceptionally suitable for practical and scalable smart farming deployment.

## 5. Reproducibility & Code

### 5.1 Project Structure

```text
Kambing/
├── Dataset/                        ← Preprocessed audio clips (2.0s .wav)
│   ├── lapar/                      ← 247 samples
│   ├── normal/                     ← 373 samples
│   └── stress/                     ← 97 samples
│
├── Rawdataset/                     ← Raw recordings (.m4a)
│   ├── lapar/
│   ├── normal/
│   └── stress/
│
├── Notebooks/
│   ├── preprocessing.ipynb         ← Audio conversion, segmentation, filtering
│   ├── eda.ipynb                   ← Exploratory data analysis & visualization
│   └── research_experiments.ipynb  ← ⭐ Full 4-experiment pipeline
│
├── Results/                        ← Auto-generated outputs
│   ├── *_history.png               ← Training curves
│   ├── *_cm.png                    ← Confusion matrices
│   ├── *_classification_report.csv ← Per-class metrics
│   └── ringkasan_seluruh_eksperimen.csv ← Aggregated results
│
├── app.py                          ← Streamlit inference application
├── best_model_dnn_final.keras      ← Saved best model weights
├── label_encoder.pkl               ← Class label encoder
├── norm_params.npy                 ← Feature normalization parameters
└── requirements.txt                ← Python dependencies
```

### 5.2 Requirements

- Python ≥ 3.10
- See `requirements.txt` for full dependency list

Key libraries:

| Library | Purpose |
|---------|---------|
| `librosa` | Audio loading, MFCC extraction |
| `tensorflow / keras` | Model definition, training |
| `scikit-learn` | Data splitting, class weights, metrics |
| `numpy`, `pandas` | Numerical computation, data management |
| `streamlit` | Web inference interface |

### 5.3 Running the Experiments

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/username/Kambing.git
cd Kambing
```

**Step 2 — Create virtual environment:**
```bash
# Windows
python -m venv .venv && .venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv && source .venv/bin/activate
```

**Step 3 — Install dependencies:**
```bash
pip install --upgrade pip && pip install -r requirements.txt
```

**Step 4 — Prepare folder structure:**
```bash
mkdir -p Dataset/lapar Dataset/normal Dataset/stress
mkdir -p Rawdataset/lapar Rawdataset/normal Rawdataset/stress Results
```

**Step 5 — Preprocessing** *(run after placing raw `.m4a` files in `Rawdataset/`)*:

Open and run all cells in `Notebooks/preprocessing.ipynb`.

**Step 6 — Run experiments:**

Open and run all cells in `Notebooks/research_experiments.ipynb`. All 4 experiments execute sequentially. Results are saved automatically to `Results/`.

**Step 7 — Launch inference application:**
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` to upload goat audio files and receive real-time condition predictions.

---

## 6. Dataset Access

The audio dataset is **not included** in this repository due to file size constraints. The dataset constitutes primary research data collected under a formal research protocol.

> [!IMPORTANT]
> **To request access to the dataset:**
>
> Please contact the corresponding author directly via email, providing your institutional affiliation and intended use case. Dataset access is granted for **academic and research purposes only**.
>
> 📧 **Contact:** [wahyukusumaw29@gmail.com](mailto:wahyukusumaw29@gmail.com)
>
> Alternatively, request access via the Google Drive link: 🔗 **[Request Dataset Access](https://drive.google.com/drive/folders/1GbyV6n6u0nKs_SCsu1AZ1Vo0HdbJR2Uh?usp=sharing)**

> [!NOTE]
> **To request the complete Google Colab Notebook:**
>
> The full, comprehensive Google Colab notebook (`research_pipeline.ipynb`) containing the entire training and optimization process is available upon request. Please email the author specifying your academic background and purpose.
>
> 📧 **Contact:** [wahyukusumaw29@gmail.com](mailto:wahyukusumaw29@gmail.com)

### Dataset Terms of Use

- Academic use only
- Proper attribution required in any publication using this dataset
- Commercial use strictly prohibited without written permission
- Redistribution without consent is not permitted

---

*All experiments are fully reproducible with `SEED = 42` applied to Python `random`, NumPy, and TensorFlow random number generators.*
