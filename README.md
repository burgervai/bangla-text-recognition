# Bangla Handwritten Word Recognition

CNN-based OCR for isolated Bangla characters with Streamlit drawing UI and MLflow tracking.

## Dataset

Download **BanglaLekha-Isolated** from [Mendeley Data](https://data.mendeley.com/datasets/hf6sf8zrkc/2).  
Extract so the structure is:
```
BanglaLekha-Isolated/
  অ/  *.png
  আ/  *.png
  ...
```

## Approach

### Preprocessing
- Grayscale, resize to 64×64
- Invert (white bg → black bg)
- Normalize to [0, 1]

### Model Architecture
**CNN (default):** 3× Conv2D→MaxPool → Flatten → Dense(256) → Dropout(0.4) → Softmax  
**Transfer (optional):** MobileNetV2 (frozen) + grayscale→RGB adapter → Dense head


## Steps to keep Python 3.14 globally and use Python 3.11 for this project

### 1. Install Python 3.11
- Download from https://www.python.org/downloads/release/python-311/
- Run the installer
- Make sure to check:
  - `Add Python 3.11 to PATH`
  - `Install launcher for all users (recommended)`

### 2. Verify Python 3.11 is available
Open PowerShell and run:
```powershell
py -3.11 --version
```
If that works, you can use `py -3.11` without changing your global Python 3.14 setup.

### 3. Create a venv for this repo
In your project folder:
```powershell
cd C:\Users\DataInsight\Downloads\bangla-ocr-assignment
py -3.11 -m venv .venv
```

### 4. Activate the new venv
PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks scripts, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
then activate again.

### 5. Upgrade pip and install requirements
With the venv active:
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```


### Training
```bash
python train.py                   # CNN, 20 epochs
python train.py --transfer        # MobileNetV2
python train.py --epochs 30 --lr 0.0005
```
Outputs: `models/model.keras`, `labels.json`

### MLflow Tracking
```bash
mlflow ui --host 0.0.0.0 --port 5000
```
Two required runs:
1. `python train.py` → CNN baseline
2. `python train.py --transfer` → MobileNetV2 transfer

Logged: model_type, epochs, lr, batch_size, img_size, num_classes, val_accuracy, val_loss + artifacts.

### Word Segmentation Strategy
1. Convert canvas RGBA → grayscale
2. Otsu thresholding → binary mask
3. Morphological dilation to merge strokes
4. Find external contours → bounding boxes
5. Filter noise (w>5, h>5), sort left→right
6. Pad each segment, resize to 64×64, invert
7. Predict each character independently, concatenate Unicode chars

### Streamlit UI
```bash
streamlit run app.py
```
- Draw one Bangla word on the 600×200 canvas
- Click **Recognize**
- See: full predicted word + per-character image + confidence %

### Docker
```bash
docker build -t bangla-ocr-app:0.1 .
docker run -p 8501:8501 bangla-ocr-app:0.1
```
Open http://localhost:8501

**Note:** Train model locally first, then copy `models/model.keras` and `labels.json` before building Docker image.

## Project Structure
```
bangla-ocr-assignment/
├── train.py
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── labels.json
├── models/
│   └── model.keras
├── artifacts/
│   └── mlflow/
└── screenshots/
    ├── streamlit_app.png
    └── mlflow_experiment.png
```

## Limitations
- Character segmentation fails on heavily connected/ligated writing
- Model trained on isolated chars; cursive word style may differ
- No matra/diacritic-aware post-processing

## Possible Improvements
- Sequence model (CRNN + CTC) for end-to-end word recognition
- Data augmentation (rotation, elastic distortion)
- Fine-tune on word-level dataset
- Vowel diacritic detection as separate classifier
