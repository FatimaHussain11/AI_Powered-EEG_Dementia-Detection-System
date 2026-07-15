# 🧠 EEG-Based Dementia Screening Tool

A full-stack ML web app that looks at resting-state EEG recordings and helps tell apart **Alzheimer's Disease (AD)**, **Frontotemporal Dementia (FTD)**, and **Healthy Controls (HC)**. The backend runs on Flask with a scikit-learn/XGBoost inference pipeline, paired with a lightweight vanilla JS/HTML/CSS frontend.

> ⚠️ **Heads up:** This is a university research project and a decision-support prototype — nothing more. It is **not** a certified medical device and should never be used for actual clinical diagnosis.

---

## Screenshots


<img width="1456" height="837" alt="home" src="https://github.com/user-attachments/assets/7a7b6638-baff-47a2-922c-a0dd237e69b2" />
<img width="1182" height="810" alt="prediction" src="https://github.com/user-attachments/assets/0591c31b-321d-4d0b-89b7-440eaa4eba31" />
<img width="1142" height="1428" alt="results" src="https://github.com/user-attachments/assets/74c14160-fafd-4a06-80b3-3c13453fe5cc" />
<img width="1184" height="564" alt="about" src="https://github.com/user-attachments/assets/9720ae8b-6e75-4934-b46d-d502040a09fe" />
<img width="854" height="838" alt="contact" src="https://github.com/user-attachments/assets/7e2fcf7d-8121-4b09-b700-874c17713ef4" />
<img width="1110" height="433" alt="disease-info" src="https://github.com/user-attachments/assets/ae2ad059-eecf-4278-a74d-8d7d9f8ce258" />
<img width="1434" height="659" alt="features" src="https://github.com/user-attachments/assets/af51f4c5-ba8f-4d38-a64a-3cd4f32f5ed3" />
<img width="1231" height="410" alt="footer" src="https://github.com/user-attachments/assets/8ed6b47f-956c-4f71-9113-a3db0ee577e3" />
<img width="1568" height="495" alt="how-it-works" src="https://github.com/user-attachments/assets/11a98b05-57dc-4982-af6d-1cfe28970f34" />
<img width="1568" height="314" alt="stats" src="https://github.com/user-attachments/assets/500e3bc4-d94d-4171-93dd-06ee40dd2b97" />

---

## 📋 Contents

- [What This Is](#-what-this-is)
- [The Dataset](#-the-dataset)
- [What It Can Do](#-what-it-can-do)
- [Pipeline Walkthrough](#-pipeline-walkthrough)
- [Repo Layout](#-repo-layout)
- [Built With](#-built-with)
- [Running It Locally](#-running-it-locally)
- [API](#-api)
- [Models](#-models)
- [Limitations](#-limitations)
- [What's Next](#-whats-next)
- [Citation](#-citation)
- [License](#-license)
- [Contact](#-contact)

---

## 🔍 What This Is

The goal here is to use machine learning on raw EEG signals to make dementia screening faster and non-invasive. A user drops an EEG file into the dashboard, chooses which classifier to run, and gets back a prediction, a confidence score, and a breakdown of probabilities across classes — all shown in an interactive results view with charts.

There are two ways to run a prediction:

| Mode | Classes | What it separates |
|---|---|---|
| **Binary** | `AD` vs `HC` | Dementia vs. Healthy |
| **Three-Class** | `AD` vs `FTD` vs `HC` | Alzheimer's vs. Frontotemporal Dementia vs. Healthy |

---

## 🗃 The Dataset

Training data comes from **[OpenNeuro ds004504](https://openneuro.org/datasets/ds004504/versions/1.0.9)**:

> **"A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG"**
> Miltiadous, A., Tzimourta, K. D., Afrantou, T., et al. (2023). *Data*, 8(6), 95.

**Quick facts:**
- **88 subjects** total — resting-state, eyes-closed recordings
- **36** Alzheimer's Disease cases
- **23** Frontotemporal Dementia cases
- **29** Healthy Controls
- Captured on a **Nihon Kohden EEG 2100** clinical system
- **19 scalp electrodes** (10–20 system: Fp1, Fp2, F3, F4, C3, C4, P3, P4, O1, O2, F7, F8, T3, T4, T5, T6, Fz, Cz, Pz) plus 2 mastoid references
- Cognitive scoring via **MMSE**
- Cleaned up with Artifact Subspace Reconstruction and ICA-based rejection (EEGLAB), packaged in BIDS format
- **CC0** licensed

Using the dataset? Cite the original paper — see [Citation](#-citation).

---

## ✨ What It Can Do

- 🧠 **Signal feature extraction** — pulls delta/theta/alpha/beta band power per channel, per epoch
- ⚡ **Quick inference** — predictions come back in seconds from a pre-trained pipeline
- 📊 **Confidence + probabilities** — every result ships with a confidence score and full per-class breakdown
- 🏥 **Plain-language context** — each prediction comes with a short, readable description of the condition
- 📄 **Exportable PDF reports** — download results as a PDF (powered by jsPDF)
- 📱 **Mobile-friendly UI** — responsive layout with proper semantic HTML/ARIA

---

## ⚙️ Pipeline Walkthrough

```
 1. Upload a Recording     →  User drops a .set (EEGLAB) file into the dashboard
 2. Pick a Model           →  Binary (AD vs HC) or Three-Class (AD vs FTD vs HC)
 3. Backend Crunches It    →  Preprocess → extract band-power features
                               → scale/select features → run classifier
 4. Get Results Back       →  Label, confidence %, per-class probabilities,
                               and a short description land back in the UI
```

**Three-class flow (the one that's actually production-ready):**

```
EEGLAB .set file
   → mne.io.read_raw_eeglab()
   → band-pass filter (0.5–30 Hz)
   → chopped into 4s epochs
   → Welch PSD per epoch/channel
   → band power (delta/theta/alpha/beta) + slowing ratio per channel  → 95 features/epoch
   → StandardScaler → Feature Selector → classifier.predict_proba()
   → epoch-level probabilities averaged into one patient-level result
```

---

## 🗂 Repo Layout

```
week 1/
├── app.py                              # Flask routes + request handling
├── predict.py                          # Real inference pipeline (features + model calls)
├── model_utils.py                      # Old/reference model-loading code
├── utils.py                            # Upload validation helpers
├── requirements.txt                    # Python deps
├── Alzheimer's.py                      # Exploratory/training script
│
├── models/                             # Saved model files (joblib .pkl)
│   ├── production_eeg_model.pkl               # Binary classifier
│   ├── production_scaler.pkl                  # Binary scaler
│   ├── production_eeg_model_3class.pkl        # Three-class classifier
│   ├── production_scaler_3class.pkl           # Three-class scaler
│   ├── production_selector_3class.pkl         # Three-class feature selector
│   └── production_label_encoder_3class.pkl    # Three-class label encoder
│
├── gsp-alzheimer-detection-main/       # Training/experiments
│   ├── main.ipynb                          # Training notebook
│   ├── features_epoched_v2.csv             # Extracted features
│   ├── heldout_test.pkl                    # Held-out test split
│   └── README.md
│
├── static/                             # Frontend assets
│   ├── script.js
│   └── style.css
│
├── templates/
│   └── index.html                          # Dashboard (Jinja2)
│
├── dataset/                            # Raw/processed EEG data (ds004504)
├── uploads/                            # Where uploaded files land at runtime
├── server open.pdf                     # Project write-up
└── __pycache__/
```

---

## 🛠 Built With

**Backend**
- [Flask](https://flask.palletsprojects.com/) + Flask-CORS
- [scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/)
- [MNE-Python](https://mne.tools/) for reading/preprocessing EEG
- [joblib](https://joblib.readthedocs.io/), NumPy, pandas, SciPy

**Frontend**
- HTML5, CSS3, vanilla JS
- [Chart.js](https://www.chartjs.org/) for probability/confidence visuals
- [jsPDF](https://github.com/parallax/jsPDF) for PDF export

**Data**
- [OpenNeuro ds004504](https://openneuro.org/datasets/ds004504/versions/1.0.9)

---

## 🚀 Running It Locally

### You'll need
- Python 3.10+
- pip

### 1. Grab the repo

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Set up a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

### 3. Install the requirements

```bash
pip install -r requirements.txt
```

### 4. Start the app

```bash
python app.py
```

Leave that terminal running — the server needs to stay alive.

### 5. Open it up

```
http://127.0.0.1:5000
```

> On Windows, from the project folder, that's:
> ```
> cd C:\Users\<you>\Downloads\week 1
> python app.py
> ```
> then hitting `http://127.0.0.1:5000` in a browser.

---

## 📡 API

### `POST /predict`

Kicks off a prediction on an uploaded EEG file.

**Form-data fields:**

| Field | Type | Required | What it is |
|---|---|---|---|
| `file` | file | ✅ | EEG recording (`.set`, `.edf`, `.csv`, `.mat`, `.txt`, `.npy`) |
| `model` | string | ✅ | `binary` or `three-class` |

**curl example:**

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -F "file=@sample_recording.set" \
  -F "model=three-class"
```

**On success (`200`):**

```json
{
  "status": "Success",
  "prediction": "Frontotemporal Dementia",
  "confidence": 88.2,
  "model": "Three-Class Classification",
  "probabilities": { "AD": 5.1, "FTD": 88.2, "HC": 6.7 },
  "description": "Frontotemporal dementia primarily affects the frontal and temporal lobes, impacting behavior, personality, and language."
}
```

**On error (`400` / `422` / `500` / `503`):**

```json
{
  "status": "Error",
  "prediction": null,
  "confidence": null,
  "model": null,
  "probabilities": {},
  "description": "",
  "message": "Descriptive error message here."
}
```

---

## 🤖 Models

| Pipeline | Input | Status |
|---|---|---|
| **Binary (AD vs HC)** | 36-dim feature vector | ⚠️ Currently placeholder/mock features — see limitations |
| **Three-Class (AD/FTD/HC)** | 95 band-power features (19 channels × 5 features) via Welch PSD, per 4s epoch | ✅ Real pipeline, ported from the training notebook |

Both were trained in `gsp-alzheimer-detection-main/main.ipynb`, saved with `joblib`, and loaded once when the app starts so inference stays fast.

---

## ⚠️ Limitations

- **The binary model is running on fake features right now.** The real graph-signal-processing feature extraction from training never got ported into `predict.py` — `extract_mock_features_binary()` just returns placeholders. **Don't trust binary-mode output** until that's fixed.
- The three-class pipeline only handles **EEGLAB `.set`** files for real predictions. Other formats (`.csv`, `.edf`, `.mat`, `.txt`, `.npy`) pass upload validation but don't have a matching feature-extraction path yet.
- This is a research prototype, not a validated clinical tool — it should never be used on its own to make a diagnosis.

---

## 🗺 What's Next

- [ ] Wire up real feature extraction for `extract_mock_features_binary()`
- [ ] Build out `.edf`, `.csv`, `.mat`, `.npy` feature-extraction paths
- [ ] Add auth for any clinical/research deployment
- [ ] Dockerize
- [ ] Add automated tests for the inference pipeline

---

## 📚 Citation

If you use this project or its dataset, please cite:

```bibtex
@article{miltiadous2023dataset,
  title={A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG},
  author={Miltiadous, Andreas and Tzimourta, Katerina D. and Afrantou, Theodora and Ioannidis, Panagiotis and Grigoriadis, Nikolaos and Tsalikakis, Dimitrios G. and Angelidis, Pantelis and Tsipouras, Markos G. and Glavas, Euripidis and Giannakeas, Nikolaos and Tzallas, Alexandros T.},
  journal={Data},
  volume={8},
  number={6},
  pages={95},
  year={2023},
  publisher={MDPI}
}
```

Dataset DOI: [10.18112/openneuro.ds004504](https://doi.org/10.18112/openneuro.ds004504)

---

## 📄 License

MIT-licensed (see [LICENSE](LICENSE)). The dataset itself (OpenNeuro ds004504) is **CC0**.

---

## 📬 Contact

Built as a university AI project.

Got questions or want to collaborate? Open an issue, or reach out through the contact form in the app.

---

<p align="center"><sub>Built with Flask, scikit-learn, MNE-Python, and way too many EEG epochs 🧠⚡</sub></p>
