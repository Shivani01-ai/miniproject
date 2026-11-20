# Document Integrity Verification and Tampering Localization

An academic AI/ML mini-project designed to detect, highlight, and report suspicious digital tampering in scanned and photographed documents (such as invoices, certificates, and marksheets) using **classical computer vision forensics**, **OCR layout heuristics**, and **deep learning inspection**.

---

## 1. Project Objectives
1. **Document Ingestion:** Enable users to upload document images (PNG, JPG, JPEG).
2. **Text Extraction:** Extract text content and bounding box coordinates using OCR.
3. **Forensic Image Analysis:** Detect compression discrepancies and noise irregularities using techniques like Error Level Analysis (ELA) and noise residual mapping.
4. **Suspicious Region Localization:** Overlay an intuitive heatmap highlighting anomalous regions directly on the document.
5. **Auditing & Reporting:** Produce an exportable PDF report summarizing forensic scores, extracted text, and cropped suspicious areas.
6. **Deep Learning Integration:** Connect a compatible deep learning tampering-detection model, with honest and graceful fallback if weights are unavailable.

---

## 2. Directory Structure

```text
Document-Integrity-verification/
│
├── app.py                  # Main Streamlit web application entry point
├── requirements.txt        # Python package dependencies
├── README.md               # Project documentation and setup guide
│
├── core/                   # Pipeline orchestration and global settings
│   └── __init__.py
│
├── forensics/              # Classical signal analysis (ELA, Noise, Sharpness)
│   └── __init__.py
│
├── ocr/                    # OCR extraction and layout sanity checking
│   └── __init__.py
│
├── models/                 # Deep learning tampering detection module
│   └── __init__.py
│
├── reporting/              # Downloadable PDF report generation
│   └── __init__.py
│
└── samples/                # Sample test documents (clean & tampered)
    └── README.md
```

---

## 3. Module Descriptions (Beginner Guide)

* **`app.py`:** Runs the web dashboard using Streamlit. Handles user interactions, file uploads, side-by-side visual comparisons, and report download triggers.
* **`core/`:** Contains central configuration (`config.py`) and the pipeline orchestrator (`pipeline.py`) that coordinates all components in sequence.
* **`forensics/`:** Implements deterministic, explainable computer vision techniques:
  * *Error Level Analysis (ELA):* Detects differences in JPEG compression levels between unaltered and digitally inserted content.
  * *Noise Variance:* Analyzes digital noise patterns to spot spliced or smoothed regions.
* **`ocr/`:** Reads text and analyzes physical layout:
  * Identifies word coordinates $(x, y, w, h)$.
  * Flags suspicious text baseline drops or unexpected character size variations.
* **`models/`:** Encapsulates the PyTorch deep learning detector. If weights are loaded, it computes patch-level tampering probabilities. If not loaded, it alerts the user and falls back safely to classical forensics.
* **`reporting/`:** Uses `fpdf2` to construct a clean, multi-page audit report for external evaluation.
* **`samples/`:** A dedicated folder to store sample clean documents and tampered test cases for testing and viva demonstrations.

---

## 4. Setup & Installation (Windows)

> **Important:** Run these commands in your Windows terminal / PowerShell when you are ready to install dependencies.

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv

# 2. Activate the virtual environment
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the web application
streamlit run app.py
```

---

## 5. Scope & Academic Ethics
- **Suspicion vs. Proof:** The tool indicates **potential anomalies and suspicious areas**. It does not guarantee authenticity or claim definitive legal proof without an original reference.
- **Extensions:** Unconstrained handwriting analysis and AI-generated text classification are reserved as future enhancements.

