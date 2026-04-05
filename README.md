# 🧠 Emotion Detection AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![ResNet50](https://img.shields.io/badge/Model-ResNet50-00897B?style=for-the-badge)

**A deep learning web app that detects human emotions from facial images in real time.**

[🚀 Live Demo](https://emotion-detection-fvjkwmkhacc8w43ah4fgjr.streamlit.app) · [📓 Notebook](./Transfer_learning_finetunning_emotion_detection_Project__1_.ipynb) · [📦 Model](./emotion_model.keras)

</div>

---

## 📌 Overview

Emotion Detection AI is a computer vision project that classifies facial expressions into **7 emotion categories** using a fine-tuned **ResNet50** convolutional neural network. The model is served through a polished **Streamlit** web application deployed on Streamlit Cloud.

Upload any face image and the model will instantly predict the dominant emotion along with a confidence score and a full probability breakdown chart.

---

## 🎭 Supported Emotions

| # | Emotion | Emoji |
|---|---------|-------|
| 0 | Angry | 😠 |
| 1 | Disgust | 🤢 |
| 2 | Fear | 😨 |
| 3 | Happy | 😄 |
| 4 | Neutral | 😐 |
| 5 | Sad | 😢 |
| 6 | Surprise | 😲 |

---

## 🏗️ Project Architecture

```
emotion-detection/
├── app.py                  # Streamlit web application
├── emotion_model.keras     # Trained model weights
├── requirements.txt        # Python dependencies
└── .streamlit/
    └── config.toml         # Streamlit theme configuration
```

---

## 🧬 Model Architecture

The model is built using **Transfer Learning + Fine-tuning** on top of **ResNet50** pretrained on ImageNet.

```
Input (224 × 224 × 3)
        │
   ┌────▼────────────────┐
   │  Data Augmentation  │  RandomFlip · RandomRotation · RandomZoom
   └────┬────────────────┘
        │
   ┌────▼────────────────┐
   │  preprocess_input   │  ResNet50-specific normalization
   └────┬────────────────┘
        │
   ┌────▼────────────────┐
   │     ResNet50        │  Pretrained on ImageNet (23.5M params)
   │  (frozen → tuned)   │  Last 20 layers unfrozen in Phase 2
   └────┬────────────────┘
        │
   GlobalAveragePooling2D
        │
     Dropout(0.5)
        │
   Dense(7, softmax)
        │
   Output: 7 emotion probabilities
```

### Training Strategy

Training was done in **two phases**:

**Phase 1 — Transfer Learning**
- Base ResNet50 fully frozen
- Only the classification head is trained
- Learning rate: `0.0001`
- Epochs: up to 15 (with early stopping)

**Phase 2 — Fine-tuning**
- Last 20 layers of ResNet50 unfrozen
- Very low learning rate: `0.00001` to preserve pretrained features
- Epochs: up to 25 (with early stopping)

---

## 📊 Dataset

**FER-2013** (Facial Expression Recognition 2013) from Kaggle  
→ [`ananthu017/emotion-detection-fer`](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)

| Split | Images |
|-------|--------|
| Train (90%) | 25,839 |
| Validation (10%) | 2,870 |
| **Total** | **28,709** |

### Class Distribution

The dataset is **heavily imbalanced** — "happy" has ~8,000 images while "disgust" has only ~436. To fix this, **inverse-frequency class weights** were applied during training:

```python
class_weights = {i: total / (n_classes * count) for i, count in enumerate(class_counts)}
```

This ensures rare emotions like Disgust and Fear are not ignored by the model.

---

## ⚙️ Key Training Improvements

| Technique | Why |
|-----------|-----|
| **Class Weighting** | Prevents the model from being biased toward majority classes (Happy) |
| **Early Stopping** | Stops training when `val_accuracy` stops improving (patience=4) |
| **ModelCheckpoint** | Saves the best model automatically during training |
| **ReduceLROnPlateau** | Halves LR when `val_loss` plateaus — prevents getting stuck |
| **Dropout 0.5** | Reduces overfitting in the classification head |
| **RandomZoom Augmentation** | Helps generalize to different face sizes and crops |
| **Low Fine-tuning LR** | `1e-5` instead of `1e-3` — gently adjusts ResNet weights without destroying them |

---

## 🖥️ Web Application

Built with **Streamlit** and styled with custom CSS for a dark, modern aesthetic.

**Features:**
- Upload JPG, PNG, or WEBP images
- Instant emotion prediction with confidence score
- Animated confidence bar color-coded per emotion
- Runner-up emotion shown when second prediction > 12%
- Interactive Plotly bar chart of all 7 emotion probabilities

**Preprocessing in the app matches training exactly:**
```python
from tensorflow.keras.applications.resnet import preprocess_input

img = image.resize((224, 224))
img_arr = np.array(img, dtype=np.float32)
img_arr = preprocess_input(img_arr)   # ResNet50-specific normalization
img_arr = np.expand_dims(img_arr, axis=0)
```

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/OmarEssam-20/emotion-detection.git
cd emotion-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit>=1.32.0
tensorflow>=2.16.0
numpy>=1.24.0
Pillow>=10.0.0
plotly>=5.18.0
```

---

## ☁️ Deployment

Deployed on **Streamlit Cloud** with Git LFS used to handle the model file (92 MB).

**Repo structure for deployment:**
```
├── app.py
├── requirements.txt
├── emotion_model.keras     ← tracked with Git LFS
└── .streamlit/
    └── config.toml
```

---

## 📓 Notebook

The full training pipeline is in [`Transfer_learning_finetunning_emotion_detection_Project__1_.ipynb`](./Transfer_learning_finetunning_emotion_detection_Project__1_.ipynb).

It covers:
- Kaggle dataset download
- Data loading with `image_dataset_from_directory`
- Class weight computation
- Model building (ResNet50 + custom head)
- Phase 1: Transfer learning
- Phase 2: Fine-tuning
- Per-class evaluation with classification report
- Model saving and auto-download

---

## ⚠️ Important Note on Label Order

`image_dataset_from_directory` assigns integer labels **alphabetically** by folder name. The correct class order for this dataset is:

```python
CLASS_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
#   index:        0         1        2        3         4         5        6
```

> Note: `Neutral` (index 4) comes before `Sad` (index 5) and `Surprise` (index 6) alphabetically. Using the wrong order would cause the model to mislabel emotions.

---

## 👤 Author

**Omar Essam**  
GitHub: [@OmarEssam-20](https://github.com/OmarEssam-20)

---

<div align="center">
  <sub>Built with TensorFlow · Streamlit · ResNet50 · FER-2013</sub>
</div>
