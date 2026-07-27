# AI-Powered Fake News Detection Using Text Classification
**Project - 1 | Summer Internship Program in AI&ML Machine Learning 2026**  
**Indian Institute of Computing and Technology (IICT)**

---

## Project Overview
This repository contains the complete, modular machine learning pipeline built from scratch to classify news articles as **Real (`0`)** or **Fake (`1`)**. It implements the **30-Day Workflow** mandated by the IICT Summer Internship curriculum without relying on black-box NLP wrappers, ensuring a thorough understanding of text preprocessing, feature vectorization, parametric/non-parametric modeling, and rigorous evaluation.

---

## Quick Start Guide

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
