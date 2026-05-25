# Facial Emotion Recognition using EfficientNet-B2 + CBAM

Deep learning project for Facial Emotion Recognition (FER) using the FER2013 dataset, combining EfficientNet-B2, CBAM attention, transfer learning, explainability techniques, and confidence calibration.

---

## Team Members

- Alaa Affori
- Zaina Abdalhaq
- Jana Jawabreh

**Instructor:** Dr. Adnan Salman  
**Course:** Advanced Topics in Machine Learning  
**University:** An-Najah National University  

---

## Project Overview

This project presents a complete machine learning pipeline for recognizing facial emotions from images using deep learning techniques.

The system classifies facial expressions into seven emotion categories:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The project includes:

- Exploratory Data Analysis (EDA)
- Data preprocessing and augmentation
- Class imbalance handling
- EfficientNet-B2 + CBAM architecture
- Model training and optimization
- Error analysis and explainability
- Confidence calibration using Temperature Scaling

---

## Dataset

Dataset used:

FER2013 Dataset from Kaggle

https://www.kaggle.com/datasets/msambare/fer2013

Dataset statistics:

| Split | Images |
|---------|---------:|
| Train | 24,406 |
| Validation | 4,303 |
| Test | 7,178 |
| Total | 35,887 |

Emotion classes:

- angry
- disgust
- fear
- happy
- neutral
- sad
- surprise

---

## Project Structure

```bash
FACIAL_EMOTION_RECOGNITION/
│
├── checkpoints/              # Saved model checkpoints
├── eda_outputs/              # EDA figures and outputs
├── exports/                  # Exported artifacts
├── fer2013_img/              # Dataset
│
├── EDA.ipynb
├── data_pipeline.ipynb
├── model_architecture.ipynb
├── 4-model-training.ipynb
├── 5-explainability.ipynb
├── 6-inference_export.ipynb
│
├── README.md
└── requirements.txt
```

---

## Methodology

The project workflow is divided into six phases:

### Phase 1 — Exploratory Data Analysis

- Dataset inspection
- Class distribution analysis
- Class imbalance visualization
- Pixel statistics
- Image size analysis

### Phase 2 — Data Pipeline

- Custom FERDataset class
- Data augmentation
- WeightedRandomSampler
- Class-weighted loss
- DataLoaders

### Phase 3 — Model Architecture

Model architecture:

```text
Input Image
      ↓
EfficientNet-B2 Backbone
      ↓
CBAM Attention Module
      ↓
Global Average Pooling
      ↓
FC(1408 → 512)
↓
BatchNorm
↓
ReLU
↓
Dropout(0.4)
↓
FC(512 → 256)
↓
BatchNorm
↓
ReLU
↓
Dropout(0.3)
↓
FC(256 → 7)
      ↓
Emotion Prediction
```

---

## Training Strategy

| Hyperparameter | Value |
|---|---:|
| Input Size | 224×224 |
| Batch Size | 64 |
| Optimizer | AdamW |
| Base Learning Rate | 3e−4 |
| Backbone Learning Rate | 3e−5 |
| Weight Decay | 1e−4 |
| Label Smoothing | 0.1 |
| Gradient Clipping | 1.0 |
| Maximum Epochs | 50 |
| Early Stopping | 10 |

Additional techniques:

- Transfer Learning
- Mixed Precision Training
- OneCycleLR Scheduler
- Weighted Sampling
- Class-weighted CrossEntropy Loss

---

## Results

| Metric | Value |
|---|---:|
| Test Accuracy | 63.86% |
| Macro-F1 | 0.6170 |
| Best Validation Macro-F1 | 0.6214 |
| ECE Before Calibration | 0.1049 |
| ECE After Calibration | 0.0610 |
| Calibration Improvement | 41.8% |

Per-class F1 scores:

| Emotion | F1 Score |
|---|---:|
| Happy | 0.8488 |
| Surprise | 0.7627 |
| Neutral | 0.6059 |
| Disgust | 0.5929 |
| Angry | 0.5652 |
| Sad | 0.4865 |
| Fear | 0.4574 |

---

## Explainability

To improve model transparency:

- Grad-CAM visualizations
- Confusion matrix analysis
- Confidence distribution analysis
- Error analysis

---

## Challenges and Limitations

- Severe class imbalance in FER2013
- Low-resolution images
- Emotion ambiguity
- Similar facial patterns between negative emotions
- Difficulty separating Fear and Sad expressions

---

## Future Work

Possible improvements:

- Vision Transformer architectures
- Ensemble learning
- Larger and more diverse datasets
- Advanced calibration methods
- Lightweight deployment optimization

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- PIL

---

## How to Run

Clone the repository:

```bash
git clone https://github.com/your-username/Facial-Emotion-Recognition-EfficientNet-CBAM.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run notebooks in order:

```bash
1. EDA.ipynb
2. data_pipeline.ipynb
3. model_architecture.ipynb
4. 4-model-training.ipynb
5. 5-explainability.ipynb
6. 6-inference_export.ipynb
```

---

## References

FER2013 Dataset:

https://www.kaggle.com/datasets/msambare/fer2013
