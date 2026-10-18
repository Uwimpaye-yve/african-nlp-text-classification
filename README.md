# Swahili News Topic Classification Using AfriBERTa

[![Streamlit App]](https://swahili-nlp-news-classification.streamlit.app)
[![Hugging Face Model]](https://huggingface.co/uyvette/swahili-news-afriberta)
[![License: MIT]](https://opensource.org/licenses/MIT)

An end-to-end Natural Language Processing (NLP) pipeline and deployed web application for categorizing Swahili news articles into 7 thematic classes using the **MasakhaNEWS** benchmark. This project implements and evaluates an n-gram statistical baseline (**TF-IDF + Logistic Regression**) against a fine-tuned Afrocentric Transformer (**AfriBERTa**).

---

## 📌 Project Overview

- **Task:** Multi-class News Topic Categorization
- **Target Language:** Swahili (`swa`)
- **Benchmark Dataset:** [MasakhaNEWS](https://huggingface.co/datasets/masakhane/masakhanews) (`masakhane/masakhanews`, `swa` split)
- **Classes (7):** `business`, `entertainment`, `health`, `politics`, `religion`, `sports`, `technology`
- **Live Web Application:** [Swahili News Topic Classifier on Streamlit Cloud](https://swahili-nlp-news-classification.streamlit.app)
- **Model Checkpoint:** [`uyvette/swahili-news-afriberta`](https://huggingface.co/uyvette/swahili-news-afriberta)

---

## 📊 Experimental Results

Both models were evaluated on the official MasakhaNEWS Swahili test split ($N = 476$).

### Overall Performance Comparison

| Model Architecture | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline:** Sub-linear TF-IDF + Logistic Regression | 81.30% | 0.8174 | 0.7256 | 0.7529 |
| **Proposed:** Fine-Tuned `castorini/afriberta_small` | **86.97%** | **0.8621** | **0.8385** | **0.8458** |

### Systematic Controlled Ablation Study

| Experiment Configuration | Input Feature | Max Length | Epochs | Learning Rate | Test Accuracy | Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline:** TF-IDF (1-2 n-grams) + Logistic Regression | Headline + Text | — | — | — | 81.30% | 0.7529 |
| **2. Ablation 1:** AfriBERTa (Headline Only) | Headline Only | 64 | 4 | $3 \times 10^{-5}$ | 83.19% | 0.7942 |
| **3. Ablation 2:** AfriBERTa (Truncated Context) | Headline + Text | 128 | 3 | $5 \times 10^{-5}$ | 85.08% | 0.8190 |
| **4. Proposed:** Fine-Tuned `castorini/afriberta_small` | Headline + Text | 256 | 4 | $3 \times 10^{-5}$ | **86.97%** | **0.8458** |

### Per-Class Performance Breakdown

| Class | Baseline F1 | AfriBERTa F1 | Absolute $\Delta$ F1 |
| :--- | :---: | :---: | :---: |
| **sports** | 0.95 | **0.97** | +0.02 |
| **health** | 0.87 | **0.90** | +0.03 |
| **politics** | 0.84 | **0.87** | +0.03 |
| **religion** | 0.80 | **0.83** | +0.03 |
| **technology** | 0.76 | **0.81** | +0.05 |
| **business** | 0.64 | **0.74** | +0.10 |
| **entertainment** | 0.41 | **0.80** | **+0.39** |

> **Key Finding:** While statistical n-grams perform adequately on dominant lexical classes like `sports`, contextual pre-training in **AfriBERTa** almost doubles performance on underrepresented minority classes such as `entertainment` (F1 jumped from 0.41 to 0.80), demonstrating the power of multilingual transfer across African languages.

---

## 🗂️ Project Repository Structure

```text
african-nlp-text-classification/
├── app/
│   └── app.py                     # Streamlit web application script
├── data/                          # Dataset splits and figures
├── notebooks/
│   └── afrisenti_pipeline.ipynb   # Full end-to-end training and evaluation notebook
├── src/                           # Preprocessing and evaluation utilities
├── .gitignore                     # Git ignore rules (prevents tracking large binaries)
├── README.md                      # Comprehensive project documentation
└── requirements.txt               # App serving dependencies

---

## ⚙️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Uwimpaye-yve/african-nlp-text-classification.git
cd african-nlp-text-classification
```

### 2. Set Up a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Streamlit Application Locally

To start the interactive classifier on your local machine:

```bash
streamlit run app/app.py
```

The application automatically downloads the fine-tuned weights from Hugging Face Hub (`uyvette/swahili-news-afriberta`), caches them in memory, and starts a local web server at `http://localhost:8501`.

---

## 🔬 Model Training & Reproducibility

The complete exploratory data analysis (EDA), baseline training, transformer fine-tuning, metric evaluation, and confusion matrix plots are contained in `notebooks/afrisenti_pipeline.ipynb`.

### Hyperparameters (AfriBERTa Fine-Tuning):
- **Base Architecture:** `castorini/afriberta_small` (RoBERTa pre-trained on 11 African languages)
- **Max Sequence Length:** 256 tokens
- **Optimizer:** AdamW
- **Learning Rate:** $3 \times 10^{-5}$
- **Weight Decay:** 0.01
- **Epochs:** 4
- **Batch Size:** 16 (per device)
- **Evaluation Metric:** Macro F1 & Accuracy

---

## 📚 References

1. Adelani, D. I., et al. (2023). *MasakhaNEWS: News Topic Classification for African Languages*. arXiv:2304.09972.
2. Ogueji, K., Zhu, Y., & Lin, J. (2021). *Small Data? No Problem! Exploring the Viability of Pretrained Multilingual Language Models for Low-Resource African Languages*. EMNLP (Findings).