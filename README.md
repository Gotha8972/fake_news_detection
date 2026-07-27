# AI-Powered Fake News Detection Using Text Classification
**Project - 1 | Summer Internship Program in AI&ML Machine Learning 2026**  
**Indian Institute of Computing and Technology (IICT)**

---

## 📌 Project Overview
This repository contains the complete, modular machine learning pipeline built from scratch to classify news articles as **Real (`0`)** or **Fake (`1`)**. It implements the **30-Day Workflow** mandated by the IICT Summer Internship curriculum without relying on black-box NLP wrappers, ensuring a thorough understanding of text preprocessing, feature vectorization, parametric/non-parametric modeling, and rigorous evaluation.

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
Make sure Python 3.8+ is installed, then run:
```powershell
pip install -r requirements.txt
```

### 2. Prepare the Dataset
By default, the script checks for `train.csv` in the root folder.
- **Kaggle Dataset**: Download from Kaggle Fake News Detection Dataset and place `train.csv` here.
- **UCI Repository / Synthetic Fallback**: If `train.csv` is not present, the script automatically generates a highly realistic synthetic benchmark dataset (`train.csv`) on the first run so you can immediately test and verify the entire workflow.

### 3. Run the Pipeline
Execute the master pipeline script:
```powershell
python fake_news_pipeline.py
```

### 4. Output & Generated Artifacts
Upon completion, the pipeline creates:
- `plots/class_distribution.png`: Bar chart of real vs. fake article counts.
- `plots/confusion_matrix_<model>.png`: High-resolution confusion matrices for **KNN**, **Logistic Regression**, **Random Forest**, and **Neural Network**.
- `plots/model_comparison_metrics.png`: Side-by-side comparison bar chart of Accuracy, Precision, Recall, and F1-score across all 4 models.
- `model_evaluation_summary.csv`: Summary table ready to be copied into your IEEE final report.

---

## 📅 30-Day Workflow Breakdown & Code Mapping

| Week | Phase | Tasks Implemented | Code Section in `fake_news_pipeline.py` |
| :--- | :--- | :--- | :--- |
| **Week 1** | **Data Loading & Cleaning** | Dataset ingestion, lowercasing, regex-based punctuation removal, manual tokenization, stopword filtering. | `week_1_workflow()`, `clean_text()` |
| **Week 2** | **Feature Engineering & EDA** | Bag-of-Words (`CountVectorizer`), TF-IDF vectorization (`TfidfVectorizer` with n-grams), and Class Distribution visualizations. | `week_2_workflow()` |
| **Week 3** | **Model Building & Training** | Implementation of **KNN** (Non-parametric), **Logistic Regression** (Parametric), **Random Forest** (Ensemble), and **MLP / Neural Net** (Deep Learning). | `week_3_workflow()` |
| **Week 4** | **Evaluation & Presentation** | Calculation of Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and cross-algorithm comparative charts. | `week_4_workflow()` |

---

## 📝 IEEE Report Submission Guidelines
When preparing your final documentation, adhere strictly to the **IEEE Standard Structure**:
1. **Introduction**: Formulate the problem statement regarding the societal impact of misinformation and how automated ML/NLP pipelines mitigate it.
2. **Dataset Description**: Detail the source (Kaggle/UCI), row count, class balance (`label`), and raw text features (`text`).
3. **Methodology**: Explain the mathematical intuition behind TF-IDF weighting and the 4 chosen classification paradigms.
4. **Results**: Present the summary table from `model_evaluation_summary.csv` along with the confusion matrices from the `plots/` folder.
5. **Discussion**: Contrast the behavior of **Parametric models** (e.g., Logistic Regression linear decision boundary, fast inference) versus **Non-Parametric models** (e.g., KNN distance metrics, sensitivity to high dimensionality) and **Ensemble/Deep Learning approaches**.
6. **Conclusion & Future Scope**: Summarize findings and propose future enhancements (e.g., Transformer embeddings like BERT, scraping live news via `NewsAPI`).
7. **Appendix**: Attach snippets from `fake_news_pipeline.py` and sample cleaned outputs.
