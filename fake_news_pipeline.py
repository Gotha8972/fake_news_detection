"""
AI-Powered Fake News Detection Pipeline
IICT Summer Internship Program in AI&ML (Project - 1)
------------------------------------------------------
This complete pipeline implements the 30-Day Workflow for Fake News Detection:
- Week 1: Data Loading, Text Cleaning (removal of stopwords, punctuation, manual/standard tokenization)
- Week 2: Feature Engineering (TF-IDF, Bag-of-Words) & Exploratory Data Analysis (EDA)
- Week 3: Model Building (KNN, Logistic Regression, Random Forest, Simple Neural Network/MLP)
- Week 4: Model Evaluation, Metrics calculation, Confusion Matrix visualizations & Comparison
"""

import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings('ignore')

# Set up visual style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('muted')

# Optional NLTK stopwords download safely
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    # Fallback standard English stopwords if NLTK download is unavailable
    STOPWORDS = {
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
        "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
        'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll",
        'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
        'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
        'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
        'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
        'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
        'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
        "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
        'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't",
        'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't",
        'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
    }


# ==============================================================================
# WEEK 1: DATA LOADING & CLEANING
# ==============================================================================

def create_sample_dataset_if_missing(filepath: str = "train.csv") -> pd.DataFrame:
    """
    Creates a synthetic realistic news dataset if 'train.csv' is not locally found.
    This allows the script to run seamlessly right out of the box for testing and verification.
    """
    if os.path.exists(filepath):
        print(f"[Week 1] Loading existing dataset from: {filepath}")
        return pd.read_csv(filepath)

    print(f"[Week 1] '{filepath}' not found locally. Generating synthetic benchmark dataset...")
    real_news = [
        "The central bank announced a 0.25 percent interest rate hike to combat rising inflation across domestic sectors.",
        "Scientists at the research institute have developed a new solar cell prototype with 30 percent higher efficiency.",
        "The national football team secured a decisive 3-1 victory in the championship quarter-final match yesterday.",
        "Space agency engineers successfully launched the next-generation weather observation satellite into low Earth orbit.",
        "Ministry of Health released new guidelines for seasonal influenza vaccination targeting elderly populations.",
        "Technology consortium unveiled an open-source framework for benchmark testing large language models safely.",
        "Global trade summit concluded with bilateral agreements signed between participating member nations.",
        "University researchers published a peer-reviewed study detailing deep sea marine biodiversity preservation.",
        "Stock markets closed slightly higher following positive quarterly corporate earnings reports from major retail chains.",
        "Public transit authority confirmed the completion of the new subway extension connecting downtown to the airport.",
        "Researchers at MIT published a groundbreaking paper on quantum computing error correction techniques.",
        "The government approved a new infrastructure bill allocating billions for bridge and highway repairs nationwide.",
        "International climate conference participants agreed on new carbon emission reduction targets for 2030.",
        "Local hospital chain reported a fifteen percent decrease in emergency room wait times after staffing reforms.",
        "The national census bureau released updated population statistics showing urban migration trends continuing.",
        "Agricultural ministry introduced subsidized crop insurance programs for smallholder farmers in drought regions.",
        "Police department announced a community policing initiative resulting in a twenty percent drop in petty crime.",
        "Automotive manufacturer revealed plans for a new electric vehicle assembly plant creating thousands of jobs.",
        "The education department launched a digital literacy program targeting rural schools with limited internet access.",
        "Federal aviation authority cleared a new commercial drone delivery service for suburban residential areas."
    ] * 25  # 500 real articles

    fake_news = [
        "SHOCKING SECRET: Secret government cabal uses tap water to control citizen minds! Click here before banned!",
        "Alien spaceship discovered buried under Antarctic ice shelf by whistleblowers! NASA tries to cover up!",
        "MIRACLE CURE: Eating raw garlic mixed with motor oil cures all known illnesses overnight! Doctors furious!",
        "Celebrity secretly admits on live broadcast that Earth is completely flat and gravity is an illusion!",
        "BREAKING: Time traveler from the year 2099 arrives with urgent warning about robotic cat uprising!",
        "Ancient Egyptian pyramids were actually power plants built by extraterrestrials from Mars, new leak shows!",
        "Billionaire masterminds plot to replace all clouds with artificial holographic projectors next month!",
        "UNBELIEVABLE: Man drinks only soda for five years and becomes immortal with super physical strength!",
        "Secret underground laboratory breeds giant flying reptiles to replace traditional postal delivery drones!",
        "EXPOSED: Popular smartphone apps transmit subconscious hypnotic frequencies while you sleep at night!",
        "URGENT WARNING: Scientists confirm drinking coffee backwards reverses aging by twenty years instantly!",
        "Government insiders reveal secret moon base has been operational since 1972 with alien ambassadors!",
        "BREAKING BOMBSHELL: Major tech company implanting microchips in keyboards to read your thoughts remotely!",
        "Exclusive leaked documents prove that dinosaurs never existed and fossils are manufactured in Chinese factories!",
        "WAKE UP PEOPLE: Secret world government meeting confirms plan to replace oxygen with artificial substitute!",
        "VIRAL TRUTH: Eating chocolate cake for breakfast proven by shadow scientists to boost IQ by fifty points!",
        "HIDDEN CAMERA FOOTAGE: Politicians caught shapeshifting into reptilian forms during closed door session!",
        "INSIDER LEAK: Major airline secretly uses teleportation technology but charges full ticket prices anyway!",
        "SHOCKING DISCOVERY: Ancient cave painting depicts modern smartphones proving time travel already exists!",
        "BANNED KNOWLEDGE: Whistleblower confirms the ocean is actually a giant holographic projection screen!"
    ] * 25  # 500 fake articles

    texts = real_news + fake_news
    labels = [0] * len(real_news) + [1] * len(fake_news)  # 0: Real, 1: Fake

    df = pd.DataFrame({'text': texts, 'label': labels})
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv(filepath, index=False)
    print(f"[Week 1] Synthetic dataset created and saved to '{filepath}' ({len(df)} rows).")
    return df


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Cleans raw news article text:
    1. Converts characters to lowercase.
    2. Removes URLs, special characters, numbers, and punctuation.
    3. Tokenizes and removes common English stopwords.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase conversion
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 3. Remove punctuation and non-alphabetic characters (keep only a-z and spaces)
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 4. Manual / basic whitespace tokenization
    tokens = text.split()

    # 5. Stopwords filtering
    if remove_stopwords:
        tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 1]

    return ' '.join(tokens)


def week_1_workflow(data_path: str = "train.csv") -> pd.DataFrame:
    """Executes Week 1 tasks: Data collection, inspection, and cleaning."""
    print("\n" + "=" * 70)
    print("WEEK 1: DATA COLLECTION, CLEANING & PREPROCESSING")
    print("=" * 70)

    data = create_sample_dataset_if_missing(data_path)
    print(f"Dataset Shape: {data.shape}")
    print(f"Class Distribution:\n{data['label'].value_counts().rename({0: 'Real (0)', 1: 'Fake (1)'})}")

    # Check and drop null values if any
    null_count = data['text'].isnull().sum()
    if null_count > 0:
        print(f"Dropping {null_count} null entries...")
        data = data.dropna(subset=['text']).reset_index(drop=True)

    print("\nApplying text cleaning pipeline (lowercasing, punctuation & stopword removal)...")
    data['cleaned_text'] = data['text'].apply(lambda x: clean_text(x, remove_stopwords=True))

    print("\nSample Cleaned Comparison:")
    for idx in range(min(2, len(data))):
        print(f"--- Sample {idx + 1} [Label: {'Fake' if data['label'].iloc[idx] == 1 else 'Real'}] ---")
        print(f"Original : {data['text'].iloc[idx][:120]}...")
        print(f"Cleaned  : {data['cleaned_text'].iloc[idx][:120]}...\n")

    return data


# ==============================================================================
# WEEK 2: FEATURE ENGINEERING & EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

def week_2_workflow(data: pd.DataFrame, max_features: int = 5000):
    """
    Executes Week 2 tasks: Feature Engineering (TF-IDF, Bag-of-Words) and EDA.
    Returns X_train, X_test, y_train, y_test, vectorizer, and feature matrices.
    """
    print("\n" + "=" * 70)
    print("WEEK 2: FEATURE ENGINEERING & EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    # 1. Exploratory Data Analysis (Plotting Class Distribution)
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x='label', data=data, palette=['#2b5c8f', '#d95f02'])
    ax.set_xticklabels(['Real News (0)', 'Fake News (1)'])
    plt.title('Dataset Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('News Category', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.tight_layout()
    plt.savefig('plots/class_distribution.png', dpi=300)
    plt.close()
    print("Saved EDA Plot: 'plots/class_distribution.png'")

    # 2. Feature Extraction: Bag-of-Words (CountVectorizer) for comparison
    print("\nExtracting Bag-of-Words (BoW) representation...")
    bow_vectorizer = CountVectorizer(max_features=max_features)
    X_bow = bow_vectorizer.fit_transform(data['cleaned_text'])
    print(f"BoW Feature Matrix Shape: {X_bow.shape}")

    # 3. Feature Extraction: TF-IDF (TfidfVectorizer) - Primary feature for models
    print("Extracting TF-IDF (Term Frequency-Inverse Document Frequency) representation...")
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_tfidf = tfidf_vectorizer.fit_transform(data['cleaned_text'])
    print(f"TF-IDF Feature Matrix Shape: {X_tfidf.shape} (using Unigrams & Bigrams)")

    # 4. Train-Test Split (80% Train, 20% Test)
    X = X_tfidf
    y = data['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain Set Shape: {X_train.shape[0]} samples")
    print(f"Test Set Shape : {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test, tfidf_vectorizer


# ==============================================================================
# WEEK 3: MODEL BUILDING & TRAINING
# ==============================================================================

def week_3_workflow(X_train, y_train):
    """
    Executes Week 3 tasks: Builds and initializes the four required classification algorithms:
    1. KNN (Non-Parametric)
    2. Logistic Regression (Parametric)
    3. Random Forest (Ensemble)
    4. Simple Neural Network (Deep Learning / MLPClassifier)
    """
    print("\n" + "=" * 70)
    print("WEEK 3: MODEL BUILDING & TRAINING")
    print("=" * 70)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5, metric='minkowski'),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42, early_stopping=True)
    }

    trained_models = {}
    for name, model in models.items():
        print(f"Training [{name}]...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f" -> Completed training [{name}].")

    return trained_models


# ==============================================================================
# WEEK 4: EVALUATION, VISUALIZATION & COMPARISON
# ==============================================================================

def week_4_workflow(trained_models, X_test, y_test):
    """
    Executes Week 4 tasks: Evaluates all models, prints comprehensive metrics,
    and generates confusion matrix & comparison plots for report submission.
    """
    print("\n" + "=" * 70)
    print("WEEK 4: MODEL EVALUATION, VISUALIZATION & COMPARISON")
    print("=" * 70)

    results = []

    # 1. Evaluate metrics for each model
    for name, model in trained_models.items():
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted', zero_division=0)
        rec = recall_score(y_test, preds, average='weighted', zero_division=0)
        f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })

        print(f"\n-------------------------------------------------------------")
        print(f"Model: {name}")
        print(f"-------------------------------------------------------------")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, preds, target_names=['Real (0)', 'Fake (1)']))

        # Save Confusion Matrix plot
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Real (0)', 'Fake (1)'],
                    yticklabels=['Real (0)', 'Fake (1)'])
        plt.title(f'Confusion Matrix: {name}', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()

        safe_filename = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        cm_path = f'plots/confusion_matrix_{safe_filename}.png'
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f" -> Saved Confusion Matrix plot to '{cm_path}'")

    # 2. Summary Comparison DataFrame
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("FINAL SUMMARY COMPARISON TABLE")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # 3. Model Comparison Bar Plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.2

    plt.bar(x - 1.5 * width, results_df['Accuracy'], width, label='Accuracy', color='#2b5c8f')
    plt.bar(x - 0.5 * width, results_df['Precision'], width, label='Precision', color='#41b6c4')
    plt.bar(x + 0.5 * width, results_df['Recall'], width, label='Recall', color='#2ca25f')
    plt.bar(x + 1.5 * width, results_df['F1-Score'], width, label='F1-Score', color='#d95f02')

    plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
    plt.ylabel('Score (0.0 to 1.0)', fontsize=12, fontweight='bold')
    plt.title('Performance Comparison Across All 4 Algorithms', fontsize=14, fontweight='bold')
    plt.xticks(x, results_df['Model'], fontsize=10)
    plt.ylim(0, 1.15)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('plots/model_comparison_metrics.png', dpi=300)
    plt.close()
    print(" -> Saved Overall Model Comparison Plot to 'plots/model_comparison_metrics.png'")

    # Save results CSV for report inclusion
    results_df.to_csv('model_evaluation_summary.csv', index=False)
    print(" -> Saved Evaluation Table to 'model_evaluation_summary.csv'")

    return results_df


# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================

def main():
    print("Starting AI-Powered Fake News Detection Pipeline...")

    # Week 1
    data = week_1_workflow("train.csv")

    # Week 2
    X_train, X_test, y_train, y_test, vectorizer = week_2_workflow(data)

    # Week 3
    trained_models = week_3_workflow(X_train, y_train)

    # Week 4
    results_df = week_4_workflow(trained_models, X_test, y_test)

    # Save trained models and vectorizer for Streamlit app
    os.makedirs("models", exist_ok=True)
    for name, model in trained_models.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        with open(f"models/{safe_name}.pkl", "wb") as f:
            pickle.dump(model, f)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("\n -> All trained models and vectorizer saved to 'models/' directory.")

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUCCESSFULLY COMPLETED!")
    print("All outputs, plots, and evaluation metrics are saved and ready for the IEEE report.")
    print("=" * 70)


if __name__ == "__main__":
    main()
