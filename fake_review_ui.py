import streamlit as st
import pandas as pd
import numpy as np
import re
import os
from lime.lime_text import LimeTextExplainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

# ------------------------
# CONFIG
# ------------------------
MODEL_PATH = "best_model.h5"
MAX_WORDS = 20000
MAX_LEN = 200
EMBED_DIM = 128

# ------------------------
# Helper Functions
# ------------------------

def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\S+|https\S+", " url ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def build_model():
    model = Sequential()
    model.add(Embedding(input_dim=MAX_WORDS, output_dim=EMBED_DIM, input_length=MAX_LEN))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

def lime_explain(model, tokenizer, text):
    explainer = LimeTextExplainer(class_names=["Original", "AI Generated"])

    def pred_fn(samples):
        seq = tokenizer.texts_to_sequences(samples)
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        preds = model.predict(padded)
        return np.hstack([1 - preds, preds])

    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=pred_fn,
        num_features=10
    )
    return exp

# ------------------------
# UI DESIGN
# ------------------------
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🧠",
    layout="wide"
)

# ---- Session state init ----
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None

st.title("🧠 AI-Generated Review Detector")
st.caption("Bi-directional RNN + LIME Explainability")

uploaded = st.sidebar.file_uploader("📌 Upload Dataset CSV", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    required_cols = ["label", "text_", "category", "rating"]
    if not all(c in df.columns for c in required_cols):
        st.error(f"Dataset must include: {required_cols}")
        st.stop()

    # Map labels
    label_map = {"CG": 1, "OR": 0}
    df["label"] = df["label"].map(label_map)
    df["text_"] = df["text_"].astype(str).apply(clean_text)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ------------------------
    # Filters
    # ------------------------
    st.sidebar.subheader("Filter before training")

    selected_cats = st.sidebar.multiselect(
        "Select Categories to include",
        options=df["category"].unique(),
        default=list(df["category"].unique())
    )

    selected_ratings = st.sidebar.multiselect(
        "Select Ratings",
        options=sorted(df["rating"].unique()),
        default=sorted(df["rating"].unique())
    )

    df_filtered = df[df["category"].isin(selected_cats)]
    df_filtered = df_filtered[df_filtered["rating"].isin(selected_ratings)]

    if df_filtered.empty:
        st.warning("No results found with filters.")
        st.stop()

    st.write(f"📂 Filtered dataset size: **{len(df_filtered)} records**")

    texts = df_filtered["text_"].tolist()
    labels = df_filtered["label"].values

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_txt)

    X_train = pad_sequences(tokenizer.texts_to_sequences(X_train_txt), maxlen=MAX_LEN)
    X_test = pad_sequences(tokenizer.texts_to_sequences(X_test_txt), maxlen=MAX_LEN)

    # ------------------------
    # Model buttons
    # ------------------------
    st.sidebar.divider()

    if st.sidebar.button("🟦 Train New Model"):
        model = build_model()
        with st.spinner("Training model..."):
            model.fit(
                X_train,
                y_train,
                validation_split=0.2,
                epochs=5,
                batch_size=64,
                verbose=1
            )
        model.save(MODEL_PATH)
        st.session_state.model = model
        st.session_state.tokenizer = tokenizer
        st.success("Model trained and saved as best_model.h5")

    if st.sidebar.button("🟩 Load Saved Model"):
        if os.path.exists(MODEL_PATH):
            model = load_model(MODEL_PATH)
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer  # tokenizer built from this dataset
            st.success("Model loaded successfully.")
        else:
            st.error("No model found. Train first.")

    # ------------------------
    # Evaluation & Prediction
    # ------------------------
    model = st.session_state.model

    if model is not None:
        tok = st.session_state.tokenizer or tokenizer

        st.subheader("📑 Model Performance")

        loss, acc = model.evaluate(X_test, y_test, verbose=0)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{acc:.4f}")
        with col2:
            st.metric("Loss", f"{loss:.4f}")

        y_pred = (model.predict(X_test) > 0.5).astype(int)

        st.code(classification_report(y_test, y_pred), language="text")

        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, y_pred))

        st.divider()
        st.subheader("🔍 Test a Review")

        user_text = st.text_area("Enter review text:")
        if user_text:
            cleaned = clean_text(user_text)
            seq = tok.texts_to_sequences([cleaned])
            pad_seq = pad_sequences(seq, maxlen=MAX_LEN)
            prob = model.predict(pad_seq)[0][0]

            label = "AI-Generated" if prob >= 0.5 else "Original"
            color = "#ffcccc" if prob >= 0.5 else "#ccffcc"

            st.markdown(
                f"""
                <div style="padding:15px;border-radius:10px;background:{color}">
                <h3 style="margin:0;">{label}</h3>
                <p>Probability: <b>{prob:.4f}</b></p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------
            # CLEAN EXPLAINABILITY
            # ------------------------
            st.write("### Explainability (Clean View)")

            exp = lime_explain(model, tok, cleaned)

            # Get weighted words
            weights = exp.as_list()

            # Separate contributions
            ai_words = [w for w, v in weights if v > 0]
            human_words = [w for w, v in weights if v < 0]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔶 Indicates AI-style")
                st.write(", ".join(ai_words[:8]) if ai_words else "None")

            with col2:
                st.markdown("#### 🔷 Indicates Human-style")
                st.write(", ".join(human_words[:8]) if human_words else "None")

            # Highlight words
            def highlight_text(text, ai_words, human_words):
                result = text.split()
                final = []

                for word in result:
                    w = word.lower().strip(".,!?")

                    if w in [a.lower() for a in ai_words]:
                        final.append(
                            f"<span style='background-color:#ffb347;padding:2px 4px;border-radius:4px;'>{word}</span>"
                        )
                    elif w in [h.lower() for h in human_words]:
                        final.append(
                            f"<span style='background-color:#6bb4ff;padding:2px 4px;border-radius:4px;'>{word}</span>"
                        )
                    else:
                        final.append(word)

                return " ".join(final)

            highlighted = highlight_text(user_text, ai_words, human_words)

            st.markdown("#### Highlighted Sentence Interpretation", unsafe_allow_html=True)
            st.markdown(
                f"<div style='font-size:18px;line-height:1.7;margin-top:10px'>{highlighted}</div>",
                unsafe_allow_html=True
            )

            
    else:
        st.info("Train or load a model to enable evaluation and prediction.")
else:
    st.info("Upload CSV to begin.")
