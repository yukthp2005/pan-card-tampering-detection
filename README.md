# Automated PAN Card Tampering & Forgery Detection System

An enterprise-grade document forensic verification portal combining **Transfer Learning (VGG16)**, **Explainable AI (Grad-CAM)**, and **Optical Character Recognition (OCR)** to detect digital forgery in Indian PAN cards with audit logging and authentication.

---

## 📌 Features

- **Synthetic Forgery Generation**: Programmatically simulates real-world tampering artifacts (patch boxes, font mismatches, Gaussian blurring, pixel noise, and JPEG compression degradation).
- **Deep Learning Classification**: Fine-tuned VGG16 convolutional neural network achieving ~87% validation accuracy on document forgery detection.
- **Explainable AI (Grad-CAM)**: Traces gradients through `block5_conv3` to generate visual thermal heatmaps, pinpointing the exact manipulated pixels.
- **OCR Format Validation**: Uses Tesseract OCR and regular expressions to extract and validate the 10-character alphanumeric Indian PAN format (`[A-Z]{5}[0-9]{4}[A-Z]{1}`).
- **Compliance Audit Logging**: SQLite database backed by Flask-SQLAlchemy tracking session timestamps, file metadata, confidence scores, and verdicts.
- **Clean Dashboard UI**: Built with Tailwind CSS for streamlined agent verification workflows.

---

## 📁 Project Structure

```text
PCTDS/
├── dataset/
│   ├── real/                  # Augmented authentic document images
│   └── tampered/              # Synthetically forged document images
├── raw_dataset/
│   ├── train/images/          # Base template images
│   └── valid/images/
├── static/
│   └── uploads/               # User scan uploads and Grad-CAM overlays
├── templates/
│   ├── index.html             # Verification dashboard & scan history
│   ├── login.html             # Agent authentication portal
│   └── register.html          # Agent account registration
├── .gitignore                 # Excluded environments, databases, and heavy weights
├── app.py                     # Flask backend, Grad-CAM pipeline & database models
├── pan_detector.keras         # Trained model weights (excluded from Git; >100 MB)
├── requirements.txt           # Python dependencies
├── setup_dataset.py           # Synthetic tampering & augmentation pipeline
├── train_model.py             # VGG16 transfer learning training script
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yukthp2005/pan-card-tampering-detection.git](https://github.com/yukthp2005/pan-card-tampering-detection.git)
cd pan-card-tampering-detection
```

### 2. Set Up Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR (Windows)
1. Download and run the Windows installer from [UB-Mannheim Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
2. Install to the default directory: `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3. If installed in a custom directory, update the path in `app.py`:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
   ```

---

## 🚀 Running the Pipeline

### Step 1: Generate Synthetic Training Data
Place base PAN templates inside `raw_dataset/train/images/` and run:
```bash
python setup_dataset.py
```
This generates 1,000 real augmented images and 1,000 synthetically forged images in `dataset/real/` and `dataset/tampered/`.

### Step 2: Train the Model
Train the VGG16 classification model:
```bash
python train_model.py
```
This trains the network over 10 epochs and exports the weights as `pan_detector.keras`.

### Step 3: Launch the Web Application
```bash
python app.py
```
Open `http://127.0.0.1:5000/` in your browser. Register an agent account, log in, and upload a document to run diagnostics.

---

## 📦 Note on Model Weights (`pan_detector.keras`)

Due to GitHub's **100 MB single-file limit**, the trained model weight file (`pan_detector.keras`, ~129 MB) is excluded from version control via `.gitignore`. You can obtain the model file by:
1. Training locally using `python setup_dataset.py` followed by `python train_model.py`.
2. Downloading the pre-trained `.keras` file directly from the **Releases** tab of this repository and placing it in the project root directory.

---

## 🛠️ Tech Stack

- **Core**: Python 3.10+
- **Deep Learning**: TensorFlow, Keras, VGG16, Grad-CAM
- **Computer Vision & OCR**: OpenCV, Tesseract OCR (`pytesseract`), NumPy
- **Backend**: Flask, Flask-SQLAlchemy, Werkzeug
- **Database**: SQLite
- **Frontend**: HTML5, Tailwind CSS
