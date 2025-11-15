import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="💬",
    layout="centered",
)

# -----------------------------------------
# 1️⃣ LOAD TRAINING DATA (train.txt)
# -----------------------------------------
@st.cache_resource
def load_and_train_model():
    df = pd.read_csv("train.txt", sep=";", header=None, names=["text", "emotion"])

    # Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["text"])
    y = df["emotion"]

    # Model
    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    return model, vectorizer

model, vectorizer = load_and_train_model()

# -----------------------------------------
# 2️⃣ UI DESIGN
# -----------------------------------------
st.markdown("""
<style>
    .main { background-color: #f3f7ff; }
    .emotion-card {
        background: white; padding: 30px; border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#1a3cff;'>💙 Emotion Detection App</h1>", unsafe_allow_html=True)

st.write("<p style='text-align:center;'>Type a message below and the AI will detect your emotion.</p>", unsafe_allow_html=True)

# -----------------------------------------
# 3️⃣ INPUT + PREDICTION
# -----------------------------------------
with st.container():
    st.markdown("<div class='emotion-card'>", unsafe_allow_html=True)

    user_input = st.text_area("📝 Enter your text here:", height=150)

    if st.button("🔮 Predict Emotion"):
        if user_input.strip():
            X = vectorizer.transform([user_input])
            prediction = model.predict(X)[0]

            st.markdown(
                f"<h3 style='text-align:center; color:#1a3cff;'>✨ Predicted Emotion: <b>{prediction}</b></h3>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("⚠️ Please enter some text.")

    st.markdown("</div>", unsafe_allow_html=True)
