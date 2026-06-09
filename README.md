# বাংলা হস্তলিখিত শব্দ চেনা — Bangla Handwritten Text Recognition

> CNN-based OCR for handwritten Bangla characters with contour-driven word segmentation, a Streamlit drawing interface, MLflow experiment tracking, and Docker deployment.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
  - [Custom CNN](#custom-cnn)
  - [MobileNetV2 Transfer Learning](#mobilenetv2-transfer-learning)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Word Segmentation Algorithm](#word-segmentation-algorithm)
- [Training](#training)
  - [Hyperparameters & Defaults](#hyperparameters--defaults)
  - [Data Augmentation](#data-augmentation)
  - [Callbacks](#callbacks)
  - [MLflow Tracking](#mlflow-tracking)
  - [Outputs](#outputs)
- [Streamlit App](#streamlit-app)
- [Installation](#installation)
  - [Python Version Requirement](#python-version-requirement)
  - [Windows (Python Launcher)](#windows-python-launcher)
  - [macOS / Linux](#macos--linux)
- [Usage](#usage)
- [Docker](#docker)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Known Limitations](#known-limitations)
- [Future Improvements](#future-improvements)

---

## Overview

This project implements an end-to-end pipeline for recognizing isolated handwritten Bangla characters. A user draws a word on a canvas; the app segments it into individual characters, runs each through a trained CNN, and concatenates the Unicode predictions back into a word.

Two model variants are available:
- **CNN** — a 4-block convolutional network trained from scratch (default)
- **MobileNetV2** — ImageNet-pretrained backbone with a grayscale adapter and a dense classification head

Both are tracked via MLflow for side-by-side comparison.

---

## How It Works

```
User draws word on canvas (800×220)
        │
        ▼
Canvas RGBA → Grayscale
        │
        ▼
Gaussian Blur → Otsu Thresholding → Morphological Close
        │
        ▼
FindContours → Filter noise (w>8, h>8) → Sort left→right
        │
        ▼
Per character: pad 12px → resize 64×64 → bitwise_not
        │
        ▼
model.predict(x) → argmax → labels[idx] → Unicode char
        │
        ▼
Concatenate chars → display word + per-char confidence
```

---

## Dataset

**BanglaLekha-Isolated** — a large-scale dataset of isolated handwritten Bangla characters.

- Source: [Mendeley Data](https://data.mendeley.com/datasets/hf6sf8zrkc/2)
- Format: PNG images organised into folders named by numeric class ID
- Each folder = one Bangla character class

**Required directory structure after extraction:**

```
BanglaLekha-Isolated/
└── Images/
    ├── 0/   *.png     ← class 0
    ├── 1/   *.png     ← class 1
    ├── 2/   *.png
    └── ...
```

Update `DATA_DIR` in `train.py` to point to your local `Images/` path:

```python
DATA_DIR = r"path/to/BanglaLekha-Isolated/Images"
```

The loader sorts class folders by integer name, encodes labels with `sklearn.LabelEncoder`, and saves the `int → Unicode char` mapping to `labels.json` (UTF-8).

**Train/Val split:** 80/20, stratified, `random_state=42`

---

## Model Architecture

### Custom CNN

Default model. Four convolutional blocks followed by global average pooling.

```
Input: (64, 64, 1)
│
├─ Conv2D(32, 3×3, same) → BatchNorm → ReLU → MaxPool(2×2)
├─ Conv2D(64, 3×3, same) → BatchNorm → ReLU → MaxPool(2×2)
├─ Conv2D(128, 3×3, same) → BatchNorm → ReLU → MaxPool(2×2)
├─ Conv2D(256, 3×3, same) → BatchNorm → ReLU
│
├─ GlobalAveragePooling2D
├─ Dense(256, relu)
├─ Dropout(0.4)
└─ Dense(num_classes, softmax)
```

### MobileNetV2 Transfer Learning

Activated with `--transfer`. A 1×1 conv adapter promotes the single grayscale channel to 3 channels before the frozen MobileNetV2 backbone.

```
Input: (64, 64, 1)
│
├─ Conv2D(3, 1×1, same)          ← grayscale → RGB adapter
├─ MobileNetV2(input_shape=(64,64,3), include_top=False, trainable=False)
├─ GlobalAveragePooling2D
├─ Dense(256, relu)
├─ Dropout(0.3)
└─ Dense(num_classes, softmax)
```

MobileNetV2 weights are frozen (`base.trainable = False`); only the head is trained.

---

## Preprocessing Pipeline

Applied to every image at load time and at inference:

| Step | Detail |
|------|--------|
| Read | `cv2.IMREAD_GRAYSCALE` |
| Resize | `64×64` pixels |
| Invert | `cv2.bitwise_not` — white background → black background |
| Normalize | `float32 / 255.0` → range `[0, 1]` |
| Shape | Expand dims to `(H, W, 1)` |

---

## Word Segmentation Algorithm

`segment_characters(gray)` in `app.py`:

1. **Gaussian Blur** — `(5,5)` kernel to reduce noise
2. **Otsu Thresholding** — `THRESH_BINARY_INV + THRESH_OTSU`, auto-computes threshold
3. **Morphological Close** — `MORPH_RECT (3,3)`, 1 iteration — closes small gaps between strokes
4. **Find contours** — `RETR_EXTERNAL`, `CHAIN_APPROX_SIMPLE`
5. **Filter** — discard bounding boxes where `w ≤ 8` or `h ≤ 8` (noise)
6. **Sort** — by `x` coordinate, left-to-right reading order
7. **Pad** — 12px border (`BORDER_CONSTANT, value=255`) on all sides
8. **Resize** — `64×64`
9. **Invert** — `bitwise_not` to match training format

Each segment is then independently classified.

---

## Training

### Hyperparameters & Defaults

| Parameter | Default | CLI flag |
|-----------|---------|----------|
| Image size | 64×64 | hardcoded |
| Batch size | 32 | hardcoded |
| Epochs | 20 | `--epochs N` |
| Learning rate | 0.001 | `--lr F` |
| Model | CNN | `--transfer` for MobileNetV2 |
| Optimizer | Adam | — |
| Loss | sparse categorical crossentropy | — |
| Val split | 20% | — |

### Data Augmentation

Applied online via `ImageDataGenerator` on training data only:

```python
rotation_range=10
width_shift_range=0.1
height_shift_range=0.1
zoom_range=0.1
```

### Callbacks

| Callback | Config |
|----------|--------|
| `EarlyStopping` | monitor `val_accuracy`, patience 5, restore best weights |
| `ReduceLROnPlateau` | monitor `val_loss`, patience 3, factor 0.5 |
| `ModelCheckpoint` | monitor `val_accuracy`, `save_best_only=True`, saves to `models/model.keras` |

### MLflow Tracking

Experiment name: `bangla-ocr`

**Logged parameters:**

```
model_type    CNN | MobileNetV2
epochs
lr
batch_size
img_size
num_classes
```

**Logged metrics:**

```
train_accuracy
train_loss
val_accuracy
val_loss
```

**Logged artifacts:**

```
accuracy_curve.png
classification_report.json
models/model.keras
labels.json
```

Run names follow the pattern: `cnn_lr0.001_ep20` / `transfer_lr0.001_ep20`

### Outputs

After training completes:

| File | Description |
|------|-------------|
| `models/model.keras` | Best checkpoint (highest val_accuracy) |
| `labels.json` | `{int: unicode_char}` mapping for all classes |
| `accuracy_curve.png` | Train vs validation accuracy per epoch |
| `classification_report.json` | Per-class precision, recall, F1, support |

---

## Streamlit App

**File:** `app.py`

The UI is a wide-layout Streamlit page with:

- A `streamlit-drawable-canvas` (800×220, white bg, black stroke, width 10)
- **Recognize Word** button — triggers segmentation + prediction
- **Clear** button — calls `st.rerun()` to reset canvas

**On recognition:**

- Displays the full predicted Unicode word in a success banner
- Shows character count
- Renders a column per detected character containing:
  - Segmented character image (inverted for display, 90px wide)
  - Predicted Unicode character (large)
  - Confidence % (softmax probability of the top class)
  - Expandable **Top 3** predictions with individual confidences

Model and labels are loaded once with `@st.cache_resource` / `@st.cache_data`.

---

## Installation

### Python Version Requirement

**Python 3.11 is required.** TensorFlow is incompatible with Python 3.12+ at the time of this project. If your system runs Python 3.12 or 3.14 globally, use a version-specific virtual environment — do **not** change your global Python.

---

### Windows (Python Launcher)

**1. Install Python 3.11**

Download from [python.org/downloads/release/python-311](https://www.python.org/downloads/release/python-311/).
During install, check both:
- ✅ Add Python 3.11 to PATH
- ✅ Install launcher for all users

**2. Verify the launcher can see it**

```powershell
py -3.11 --version
# Python 3.11.x
```

**3. Create a virtualenv pinned to 3.11**

```powershell
cd path\to\bangla-ocr-assignment
py -3.11 -m venv .venv
```

**4. Activate**

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate again.

**5. Install dependencies**

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### macOS / Linux

```bash
# Install Python 3.11 (e.g. via pyenv or brew)
pyenv install 3.11.9
pyenv local 3.11.9

# Or with brew:
brew install python@3.11

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Train — CNN (default)

```bash
python train.py
```

### Train — MobileNetV2 transfer learning

```bash
python train.py --transfer
```

### Train — custom hyperparameters

```bash
python train.py --epochs 30 --lr 0.0005
python train.py --transfer --epochs 15 --lr 0.0001
```

### Run the Streamlit app

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### Launch MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Opens at `http://localhost:5000`

To compare both model variants, run both training jobs first:

```bash
python train.py              # CNN baseline
python train.py --transfer   # MobileNetV2
```

Then open the MLflow UI and select both runs to compare params and metrics side-by-side.

---

## Docker

> **Prerequisite:** train the model locally first. The Docker image copies `models/model.keras` and `labels.json` at build time — these must exist before building.

**Dockerfile summary:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get install -y libgl1 libglib2.0-0    # OpenCV system deps
COPY requirements.txt → pip install
COPY app.py, labels.json, models/
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build:**

```bash
docker build -t bangla-ocr-app:0.1 .
```

**Run:**

```bash
docker run -p 8501:8501 bangla-ocr-app:0.1
```

Open `http://localhost:8501`

---

## Project Structure

```
bangla-ocr-assignment/
│
├── train.py                   # Training pipeline (CNN + transfer)
├── app.py                     # Streamlit inference app
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container definition
│
├── labels.json                # {int: unicode_char} class map (generated)
├── accuracy_curve.png         # Training curve (generated)
├── classification_report.json # Per-class metrics (generated)
├── mlflow.db                  # MLflow SQLite backend
│
└── models/
    └── model.keras            # Best trained model checkpoint (generated)
```

---

## Dependencies

```
tensorflow                   # Model training & inference
mlflow                       # Experiment tracking
streamlit                    # Web UI
streamlit-drawable-canvas    # Freehand drawing widget
opencv-python-headless       # Image processing & segmentation
scikit-learn                 # LabelEncoder, classification_report, train_test_split
numpy                        # Array ops
Pillow                       # Image display in Streamlit
```

System packages required inside Docker (and needed locally on Linux): `libgl1`, `libglib2.0-0`

---

## Known Limitations

- **Segmentation fails on ligated writing** — contour-based splitting assumes clear gaps between characters. Heavily cursive or connected Bangla writing breaks this assumption.
- **Isolated char training vs. word-context inference** — the model is trained on isolated characters but applied to characters segmented from words; style distribution mismatch degrades accuracy.
- **No matra / diacritic awareness** — vowel signs (matras) attached above or below a consonant are treated as separate contours and classified independently, often incorrectly.
- **Canvas stroke style dependency** — performance depends on drawing stroke width and style matching the training distribution.

---

## Future Improvements

| Improvement | Detail |
|-------------|--------|
| CRNN + CTC | End-to-end sequence model; eliminates explicit segmentation |
| Elastic distortion augmentation | Better generalisation over handwriting variation |
| Fine-tune on word-level dataset | Reduces train/inference distribution gap |
| Vowel diacritic classifier | Separate model to detect and attach matras post-prediction |
| Unfreeze MobileNetV2 head layers | Fine-tune top N layers for domain adaptation |
| Confidence threshold + rejection | Output `?` for low-confidence segments instead of a wrong character |

---

## License

MIT
