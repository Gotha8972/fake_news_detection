"""
Streamlit App - AI-Powered Fake News Detection
IICT Summer Internship Program in AI&ML (Project - 1)
-----------------------------------------------------
A premium, interactive web dashboard for:
  - Live fake news prediction on user-inputted articles
  - Model performance comparison & visualization
  - Exploratory Data Analysis with interactive charts
  - Pipeline walkthrough & educational content
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FakeScope AI · Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS for premium glassmorphism look
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main container */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0b24 0%, #1a1a3e 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}

section[data-testid="stSidebar"] .stMarkdown {
    color: #e0e0ff;
}

/* Glass card effect */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
}

/* Hero section */
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 4px;
    line-height: 1.2;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    font-weight: 300;
    margin-bottom: 24px;
    letter-spacing: 0.3px;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(139, 92, 246, 0.08));
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
}

.metric-card:hover {
    border-color: rgba(99, 102, 241, 0.5);
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.15);
}

.metric-value {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.metric-label {
    font-size: 0.85rem;
    color: #94a3b8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 6px;
}

/* Prediction result cards */
.prediction-real {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(6, 182, 212, 0.08));
    border: 1px solid rgba(16, 185, 129, 0.35);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}

.prediction-fake {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(251, 146, 60, 0.08));
    border: 1px solid rgba(239, 68, 68, 0.35);
    border-radius: 16px;
    padding: 28px;
    text-align: center;
}

.prediction-label {
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 8px;
}

.prediction-confidence {
    font-size: 1rem;
    color: #94a3b8;
    font-weight: 400;
}

/* Section headers */
.section-header {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e0e0ff;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(99, 102, 241, 0.3);
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: rgba(99, 102, 241, 0.15);
    color: #818cf8;
}

/* Text input */
.stTextArea textarea {
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(99, 102, 241, 0.25) !important;
    border-radius: 12px !important;
    color: #e0e0ff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}

.stTextArea textarea:focus {
    border-color: rgba(99, 102, 241, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 10px 32px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
}

/* Divider */
.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.4), transparent);
    margin: 24px 0;
    border: none;
}

/* Scrollbar */
::-webkit-scrollbar {width: 6px;}
::-webkit-scrollbar-track {background: rgba(255,255,255,0.02);}
::-webkit-scrollbar-thumb {background: rgba(99,102,241,0.3); border-radius: 3px;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Stopwords Fallback
# ──────────────────────────────────────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords as sw
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(sw.words('english'))
except Exception:
    STOPWORDS = {
        'i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll",
        "you'd",'your','yours','yourself','yourselves','he','him','his','himself','she',"she's",
        'her','hers','herself','it',"it's",'its','itself','they','them','their','theirs',
        'themselves','what','which','who','whom','this','that',"that'll",'these','those','am',
        'is','are','was','were','be','been','being','have','has','had','having','do','does',
        'did','doing','a','an','the','and','but','if','or','because','as','until','while','of',
        'at','by','for','with','about','against','between','into','through','during','before',
        'after','above','below','to','from','up','down','in','out','on','off','over','under',
        'again','further','then','once','here','there','when','where','why','how','all','any',
        'both','each','few','more','most','other','some','such','no','nor','not','only','own',
        'same','so','than','too','very','s','t','can','will','just','don',"don't",'should',
        "should've",'now','d','ll','m','o','re','ve','y','ain','aren',"aren't",'couldn',
        "couldn't",'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',
        "haven't",'isn',"isn't",'ma','mightn',"mightn't",'mustn',"mustn't",'needn',"needn't",
        'shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',"weren't",'won',"won't",
        'wouldn',"wouldn't",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercases, strips URLs/punctuation, removes stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 1]
    return ' '.join(tokens)


@st.cache_resource(show_spinner="🧠 Training models on dataset…")
def load_or_train_models():
    """
    Loads pre-trained pickle models if available from the pipeline run,
    otherwise trains fresh models from the dataset (or synthetic data).
    Returns: (trained_models_dict, vectorizer, results_df, data, X_test, y_test)
    """
    models_dir = "models"
    model_files = {
        "KNN": "knn.pkl",
        "Logistic Regression": "logistic_regression.pkl",
        "Random Forest": "random_forest.pkl",
        "Neural Network (MLP)": "neural_network_mlp.pkl",
    }
    vectorizer_file = os.path.join(models_dir, "tfidf_vectorizer.pkl")

    # Try to load pre-trained models
    if os.path.exists(vectorizer_file) and all(
        os.path.exists(os.path.join(models_dir, f)) for f in model_files.values()
    ):
        with open(vectorizer_file, "rb") as f:
            vectorizer = pickle.load(f)
        trained = {}
        for name, fname in model_files.items():
            with open(os.path.join(models_dir, fname), "rb") as f:
                trained[name] = pickle.load(f)

        # Still need data for EDA tabs
        data = _load_dataset()
        data['cleaned_text'] = data['text'].apply(clean_text)
        X = vectorizer.transform(data['cleaned_text'])
        y = data['label'].values
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        results = _evaluate_models(trained, X_test, y_test)
        return trained, vectorizer, results, data, X_test, y_test

    # Otherwise train fresh
    data = _load_dataset()
    data['cleaned_text'] = data['text'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(data['cleaned_text'])
    y = data['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5, metric='minkowski'),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42, early_stopping=True),
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    results = _evaluate_models(trained, X_test, y_test)
    return trained, vectorizer, results, data, X_test, y_test


def _load_dataset() -> pd.DataFrame:
    """Loads train.csv or generates synthetic data."""
    if os.path.exists("train.csv"):
        return pd.read_csv("train.csv")

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
        "Federal aviation authority cleared a new commercial drone delivery service for suburban residential areas.",
    ] * 25

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
        "BANNED KNOWLEDGE: Whistleblower confirms the ocean is actually a giant holographic projection screen!",
    ] * 25

    texts = real_news + fake_news
    labels = [0] * len(real_news) + [1] * len(fake_news)
    df = pd.DataFrame({"text": texts, "label": labels})
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df.to_csv("train.csv", index=False)
    return df


def _evaluate_models(trained_models, X_test, y_test) -> pd.DataFrame:
    """Evaluates all trained models and returns results DataFrame."""
    results = []
    for name, model in trained_models.items():
        preds = model.predict(X_test)
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, preds, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_test, preds, average="weighted", zero_division=0),
        })
    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────────────
# Load Everything
# ──────────────────────────────────────────────────────────────────────────────
trained_models, vectorizer, results_df, data, X_test, y_test = load_or_train_models()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; margin-bottom: 20px;">
        <div style="font-size: 3rem; margin-bottom: 4px;">🔍</div>
        <div style="font-size: 1.5rem; font-weight: 800;
            background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;">FakeScope AI</div>
        <div style="font-size: 0.75rem; color: #64748b; letter-spacing: 2px; text-transform: uppercase; margin-top:4px;">
            Fake News Detection System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Home & Predict", "📊 Model Comparison", "🔬 EDA & Insights", "📖 How It Works"],
        label_visibility="collapsed",
    )

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="padding: 16px;">
        <div style="font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px;">Dataset Stats</div>
        <div style="color: #e0e0ff; font-size: 0.9rem;">
            📄 <b>{:,}</b> articles<br>
            ✅ <b>{:,}</b> real &nbsp;|&nbsp; ❌ <b>{:,}</b> fake<br>
            🧩 <b>4</b> ML algorithms trained
        </div>
    </div>
    """.format(len(data), (data['label'] == 0).sum(), (data['label'] == 1).sum()), unsafe_allow_html=True)

    st.markdown("""
    <div style="position: fixed; bottom: 20px; left: 20px; font-size: 0.7rem; color: #475569;">
        IICT Summer Internship 2026<br>AI & ML · Project 1
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Home & Predict
# ──────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home & Predict":
    st.markdown('<div class="hero-title">FakeScope AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Paste any news article below and our ensemble of 4 ML algorithms will classify it as <b>Real</b> or <b>Fake</b> in real-time.</div>', unsafe_allow_html=True)

    # Quick stats row
    best = results_df.loc[results_df['Accuracy'].idxmax()]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{best["Accuracy"]:.1%}</div><div class="metric-label">Best Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{best["F1-Score"]:.1%}</div><div class="metric-label">Best F1 Score</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">4</div><div class="metric-label">ML Algorithms</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{len(data):,}</div><div class="metric-label">Training Articles</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Prediction section
    st.markdown('<div class="section-header">🔍 Analyze an Article</div>', unsafe_allow_html=True)

    col_input, col_options = st.columns([3, 1])
    with col_input:
        user_input = st.text_area(
            "Paste the news article text here:",
            height=180,
            placeholder="e.g. Scientists have developed a breakthrough quantum computing chip that can perform calculations millions of times faster...",
        )
    with col_options:
        selected_model = st.selectbox(
            "Choose model:",
            list(trained_models.keys()),
            index=1,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 Analyze Article", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        all_models_check = st.checkbox("Compare all models", value=False)

    if predict_btn and user_input.strip():
        cleaned = clean_text(user_input)
        features = vectorizer.transform([cleaned])

        if all_models_check:
            st.markdown('<div class="section-header">📋 Multi-Model Comparison</div>', unsafe_allow_html=True)
            cols = st.columns(len(trained_models))
            for i, (name, model) in enumerate(trained_models.items()):
                pred = model.predict(features)[0]
                try:
                    proba = model.predict_proba(features)[0]
                    confidence = max(proba) * 100
                except AttributeError:
                    confidence = 0.0

                label_text = "✅ Real News" if pred == 0 else "❌ Fake News"
                card_class = "prediction-real" if pred == 0 else "prediction-fake"
                color = "#10b981" if pred == 0 else "#ef4444"

                with cols[i]:
                    st.markdown(f"""
                    <div class="{card_class}">
                        <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 8px;">{name}</div>
                        <div class="prediction-label" style="color: {color};">{label_text}</div>
                        <div class="prediction-confidence">Confidence: {confidence:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            model = trained_models[selected_model]
            pred = model.predict(features)[0]
            try:
                proba = model.predict_proba(features)[0]
                confidence = max(proba) * 100
            except AttributeError:
                confidence = 0.0

            label_text = "✅ REAL NEWS" if pred == 0 else "❌ FAKE NEWS"
            card_class = "prediction-real" if pred == 0 else "prediction-fake"
            color = "#10b981" if pred == 0 else "#ef4444"

            st.markdown(f"""
            <div class="{card_class}">
                <div class="prediction-label" style="color: {color}; font-size: 2.5rem;">{label_text}</div>
                <div class="prediction-confidence" style="font-size: 1.15rem; margin-top: 8px;">
                    Model: <b>{selected_model}</b> &nbsp;·&nbsp; Confidence: <b>{confidence:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Show cleaned text
        with st.expander("🧹 View Cleaned Text (after preprocessing)"):
            st.code(cleaned, language="text")

    elif predict_btn and not user_input.strip():
        st.warning("⚠️ Please paste some article text first.")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Model Comparison
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Comparison":
    st.markdown('<div class="hero-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Comprehensive evaluation of all 4 classification algorithms on the test set.</div>', unsafe_allow_html=True)

    # Metric overview row
    cols = st.columns(len(results_df))
    colors_list = ["#818cf8", "#c084fc", "#f472b6", "#fb923c"]
    for i, row in results_df.iterrows():
        with cols[i]:
            st.markdown(f"""
            <div class="metric-card" style="border-color: {colors_list[i]}40;">
                <div style="font-size: 0.75rem; color: {colors_list[i]}; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">{row['Model']}</div>
                <div class="metric-value" style="background: {colors_list[i]}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{row['Accuracy']:.1%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Metrics Comparison", "🔥 Confusion Matrices", "📋 Detailed Report"])

    with tab1:
        # Grouped bar chart
        fig = go.Figure()
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        bar_colors = ["#6366f1", "#8b5cf6", "#06b6d4", "#f59e0b"]
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df["Model"],
                y=results_df[metric],
                marker_color=bar_colors[i],
                marker_line_width=0,
                text=[f"{v:.2%}" for v in results_df[metric]],
                textposition="outside",
                textfont=dict(size=11, color="#e0e0ff"),
            ))

        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#e0e0ff"),
            title=dict(text="Performance Metrics Comparison", font=dict(size=18)),
            yaxis=dict(range=[0, 1.18], gridcolor="rgba(255,255,255,0.06)", title="Score"),
            xaxis=dict(title=""),
            legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
            height=500,
            margin=dict(t=80, b=60),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Radar chart
        fig_radar = go.Figure()
        for i, row in results_df.iterrows():
            values = [row["Accuracy"], row["Precision"], row["Recall"], row["F1-Score"], row["Accuracy"]]
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                name=row["Model"],
                fill="toself",
                fillcolor=f"rgba({int(colors_list[i][1:3],16)},{int(colors_list[i][3:5],16)},{int(colors_list[i][5:7],16)},0.1)",
                line=dict(color=colors_list[i], width=2),
            ))
        fig_radar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#e0e0ff"),
            title=dict(text="Radar Chart — Multi-Metric Comparison", font=dict(size=18)),
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1.05], gridcolor="rgba(255,255,255,0.08)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            ),
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
            height=500,
            margin=dict(t=80, b=80),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab2:
        cm_cols = st.columns(2)
        for i, (name, model) in enumerate(trained_models.items()):
            preds = model.predict(X_test)
            cm = confusion_matrix(y_test, preds)

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Real (0)", "Fake (1)"],
                y=["Real (0)", "Fake (1)"],
                colorscale=[[0, "#1e1b4b"], [0.5, "#4338ca"], [1, "#818cf8"]],
                text=cm,
                texttemplate="%{text}",
                textfont=dict(size=20, color="white"),
                showscale=False,
            ))
            fig_cm.update_layout(
                title=dict(text=f"{name}", font=dict(size=15, color="#e0e0ff")),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#e0e0ff"),
                xaxis=dict(title="Predicted", side="bottom"),
                yaxis=dict(title="Actual", autorange="reversed"),
                height=350,
                margin=dict(t=50, b=50, l=60, r=30),
            )
            with cm_cols[i % 2]:
                st.plotly_chart(fig_cm, use_container_width=True)

    with tab3:
        for name, model in trained_models.items():
            preds = model.predict(X_test)
            report = classification_report(y_test, preds, target_names=["Real (0)", "Fake (1)"])
            with st.expander(f"📄 {name} — Full Classification Report", expanded=False):
                st.code(report, language="text")

        st.markdown("#### 📥 Summary Table")
        st.dataframe(
            results_df.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
            }).set_properties(**{"background-color": "rgba(0,0,0,0)", "color": "#e0e0ff"}),
            use_container_width=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: EDA & Insights
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🔬 EDA & Insights":
    st.markdown('<div class="hero-title">Data Exploration</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Visual analysis of the training dataset — class balance, text length distributions, and top features.</div>', unsafe_allow_html=True)

    tab_dist, tab_len, tab_feat = st.tabs(["📊 Class Distribution", "📏 Text Length Analysis", "🔤 Top TF-IDF Features"])

    with tab_dist:
        fig_dist = go.Figure()
        counts = data['label'].value_counts().sort_index()
        fig_dist.add_trace(go.Bar(
            x=["Real News (0)", "Fake News (1)"],
            y=counts.values,
            marker=dict(
                color=["#06b6d4", "#f472b6"],
                line=dict(width=0),
            ),
            text=counts.values,
            textposition="outside",
            textfont=dict(size=14, color="#e0e0ff"),
        ))
        fig_dist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#e0e0ff"),
            title=dict(text="Class Distribution in Dataset", font=dict(size=18)),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Count"),
            xaxis=dict(title=""),
            height=450,
            margin=dict(t=80, b=60),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Donut chart
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Real", "Fake"],
            values=counts.values,
            hole=0.6,
            marker=dict(colors=["#06b6d4", "#f472b6"], line=dict(color="#0f0c29", width=3)),
            textfont=dict(size=14, color="#e0e0ff"),
            textinfo="percent+label",
        )])
        fig_donut.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#e0e0ff"),
            title=dict(text="Class Ratio", font=dict(size=16)),
            height=400,
            margin=dict(t=60, b=40),
            showlegend=False,
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with tab_len:
        if 'cleaned_text' not in data.columns:
            data['cleaned_text'] = data['text'].apply(clean_text)

        data['text_length'] = data['cleaned_text'].apply(lambda x: len(x.split()))

        fig_hist = go.Figure()
        for label, color, name in [(0, "#06b6d4", "Real"), (1, "#f472b6", "Fake")]:
            subset = data[data['label'] == label]
            fig_hist.add_trace(go.Histogram(
                x=subset['text_length'],
                name=name,
                marker_color=color,
                opacity=0.7,
                nbinsx=30,
            ))
        fig_hist.update_layout(
            barmode="overlay",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#e0e0ff"),
            title=dict(text="Cleaned Text Word Count Distribution", font=dict(size=18)),
            xaxis=dict(title="Word Count", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Frequency", gridcolor="rgba(255,255,255,0.06)"),
            legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"),
            height=450,
            margin=dict(t=80, b=60),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Stats
        c1, c2 = st.columns(2)
        for col, label, color in [(c1, "Real", "#06b6d4"), (c2, "Fake", "#f472b6")]:
            subset = data[data['label'] == (0 if label == "Real" else 1)]
            avg = subset['text_length'].mean()
            med = subset['text_length'].median()
            std = subset['text_length'].std()
            with col:
                st.markdown(f"""
                <div class="glass-card" style="border-color: {color}40;">
                    <div style="color: {color}; font-weight: 700; font-size: 1.1rem; margin-bottom: 12px;">📝 {label} News</div>
                    <div style="color: #e0e0ff; font-size: 0.9rem;">
                        Mean: <b>{avg:.1f}</b> words<br>
                        Median: <b>{med:.0f}</b> words<br>
                        Std Dev: <b>{std:.1f}</b> words
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab_feat:
        feature_names = vectorizer.get_feature_names_out()

        # Get top features by TF-IDF importance per class
        real_idx = np.where(data['label'].values == 0)[0]
        fake_idx = np.where(data['label'].values == 1)[0]

        if 'cleaned_text' not in data.columns:
            data['cleaned_text'] = data['text'].apply(clean_text)

        X_all = vectorizer.transform(data['cleaned_text'])

        real_mean = np.array(X_all[real_idx].mean(axis=0)).flatten()
        fake_mean = np.array(X_all[fake_idx].mean(axis=0)).flatten()

        top_n = 15
        real_top_idx = real_mean.argsort()[-top_n:][::-1]
        fake_top_idx = fake_mean.argsort()[-top_n:][::-1]

        c1, c2 = st.columns(2)
        with c1:
            fig_real = go.Figure(go.Bar(
                y=[feature_names[i] for i in real_top_idx][::-1],
                x=[real_mean[i] for i in real_top_idx][::-1],
                orientation="h",
                marker_color="#06b6d4",
                marker_line_width=0,
            ))
            fig_real.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#e0e0ff"),
                title=dict(text="Top TF-IDF Features: Real News", font=dict(size=15)),
                xaxis=dict(title="Mean TF-IDF Score", gridcolor="rgba(255,255,255,0.06)"),
                height=500,
                margin=dict(t=60, b=40, l=120),
            )
            st.plotly_chart(fig_real, use_container_width=True)

        with c2:
            fig_fake = go.Figure(go.Bar(
                y=[feature_names[i] for i in fake_top_idx][::-1],
                x=[fake_mean[i] for i in fake_top_idx][::-1],
                orientation="h",
                marker_color="#f472b6",
                marker_line_width=0,
            ))
            fig_fake.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#e0e0ff"),
                title=dict(text="Top TF-IDF Features: Fake News", font=dict(size=15)),
                xaxis=dict(title="Mean TF-IDF Score", gridcolor="rgba(255,255,255,0.06)"),
                height=500,
                margin=dict(t=60, b=40, l=120),
            )
            st.plotly_chart(fig_fake, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: How It Works
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📖 How It Works":
    st.markdown('<div class="hero-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">A walkthrough of the 4-week AI pipeline powering this application.</div>', unsafe_allow_html=True)

    # Week 1
    st.markdown("""
    <div class="glass-card">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
            <div style="background: linear-gradient(135deg, #6366f1, #8b5cf6); width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: 800; color: white;">1</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: #e0e0ff;">Week 1 · Data Collection & Cleaning</div>
        </div>
        <div style="color: #94a3b8; line-height: 1.8; font-size: 0.95rem;">
            <b style="color:#818cf8;">Goal:</b> Transform raw messy text into clean, uniform tokens.<br>
            • <b>Lowercasing</b> — Normalizes "Government" → "government"<br>
            • <b>URL Removal</b> — Strips http/https/www links via regex<br>
            • <b>Punctuation Removal</b> — Keeps only a-z characters and spaces<br>
            • <b>Tokenization</b> — Splits text into individual words<br>
            • <b>Stopword Filtering</b> — Removes 150+ common English words (the, is, at, which…)
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Week 2
    st.markdown("""
    <div class="glass-card">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
            <div style="background: linear-gradient(135deg, #8b5cf6, #c084fc); width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: 800; color: white;">2</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: #e0e0ff;">Week 2 · Feature Engineering & EDA</div>
        </div>
        <div style="color: #94a3b8; line-height: 1.8; font-size: 0.95rem;">
            <b style="color:#c084fc;">Goal:</b> Convert text tokens into numerical feature vectors.<br>
            • <b>Bag-of-Words (BoW)</b> — Counts raw word frequencies across documents<br>
            • <b>TF-IDF</b> — Weighs words by frequency × inverse document frequency, penalizing common words<br>
            • <b>N-Grams (1,2)</b> — Captures unigrams ("tax") and bigrams ("tax cut") for context<br>
            • <b>EDA</b> — Class balance visualization, text length distributions, top-feature analysis<br>
            • <b>Train-Test Split</b> — 80% training / 20% testing with stratified sampling
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Week 3
    st.markdown("""
    <div class="glass-card">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
            <div style="background: linear-gradient(135deg, #c084fc, #f472b6); width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: 800; color: white;">3</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: #e0e0ff;">Week 3 · Model Building & Training</div>
        </div>
        <div style="color: #94a3b8; line-height: 1.8; font-size: 0.95rem;">
            <b style="color:#f472b6;">Goal:</b> Train 4 distinct ML algorithms covering different paradigms.<br><br>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                <div style="background: rgba(255,255,255,0.03); padding: 14px; border-radius: 10px; border: 1px solid rgba(99,102,241,0.15);">
                    <b style="color: #818cf8;">🔵 KNN</b> <span style="color: #64748b; font-size: 0.8rem;">(Non-Parametric)</span><br>
                    <span style="font-size: 0.85rem;">Classifies by majority vote of k=5 closest training vectors using Minkowski distance.</span>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 14px; border-radius: 10px; border: 1px solid rgba(139,92,246,0.15);">
                    <b style="color: #8b5cf6;">🟣 Logistic Regression</b> <span style="color: #64748b; font-size: 0.8rem;">(Parametric)</span><br>
                    <span style="font-size: 0.85rem;">Fits a linear decision boundary with sigmoid activation for probability output.</span>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 14px; border-radius: 10px; border: 1px solid rgba(6,182,212,0.15);">
                    <b style="color: #06b6d4;">🌲 Random Forest</b> <span style="color: #64748b; font-size: 0.8rem;">(Ensemble)</span><br>
                    <span style="font-size: 0.85rem;">Builds 100 decision trees on random feature subsets (bagging) to reduce overfitting.</span>
                </div>
                <div style="background: rgba(255,255,255,0.03); padding: 14px; border-radius: 10px; border: 1px solid rgba(244,114,182,0.15);">
                    <b style="color: #f472b6;">🧠 Neural Network / MLP</b> <span style="color: #64748b; font-size: 0.8rem;">(Deep Learning)</span><br>
                    <span style="font-size: 0.85rem;">Feed-forward network with 100 hidden neurons, backpropagation + early stopping.</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Week 4
    st.markdown("""
    <div class="glass-card">
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
            <div style="background: linear-gradient(135deg, #f472b6, #fb923c); width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; font-weight: 800; color: white;">4</div>
            <div style="font-size: 1.3rem; font-weight: 700; color: #e0e0ff;">Week 4 · Evaluation & Comparison</div>
        </div>
        <div style="color: #94a3b8; line-height: 1.8; font-size: 0.95rem;">
            <b style="color:#fb923c;">Goal:</b> Evaluate every model on unseen test data and generate publication-ready visuals.<br>
            • <b>Accuracy</b> — Overall correctness: (TP+TN) / Total<br>
            • <b>Precision</b> — Of predicted fake, how many truly fake? TP / (TP+FP)<br>
            • <b>Recall</b> — Of actual fake, how many caught? TP / (TP+FN)<br>
            • <b>F1-Score</b> — Harmonic mean of Precision & Recall<br>
            • <b>Confusion Matrix</b> — 2×2 grid of TP/TN/FP/FN per model<br>
            • <b>Bar & Radar Charts</b> — Cross-model visual comparison
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card" style="text-align: center;">
        <div style="font-size: 1.1rem; font-weight: 600; color: #818cf8; margin-bottom: 8px;">🎓 IICT Summer Internship in AI & ML — 2026</div>
        <div style="color: #64748b; font-size: 0.85rem;">Project 1: AI-Powered Fake News Detection Using Text Classification</div>
    </div>
    """, unsafe_allow_html=True)
