import streamlit as st
import joblib

st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="📩",
    layout="wide"
)

# Load files
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------- Custom CSS ----------
st.markdown("""
<style>

.main{
    background-color:#0E1117;
}

.hero{
    background: linear-gradient(
        135deg,
        #667eea,
        #764ba2,
        #ff6b6b
    );
    padding:40px;
    border-radius:25px;
    text-align:center;
    margin-bottom:30px;
}

.hero h1{
    color:white;
    font-size:50px;
}

.hero p{
    color:white;
    font-size:20px;
}

.card{
    background:#1E1E1E;
    padding:30px;
    border-radius:20px;
    box-shadow:0px 0px 20px rgba(255,255,255,0.1);
}

.metric{
    background:#262730;
    padding:20px;
    border-radius:15px;
    text-align:center;
}

textarea{
    font-size:18px !important;
}

.stButton>button{
    width:100%;
    height:60px;
    font-size:20px;
    border-radius:15px;
    border:none;
}

</style>
""", unsafe_allow_html=True)

# ---------- Hero Section ----------

st.markdown("""
<div class="hero">

<h1>📩 AI Spam Detector</h1>

<p>
Powered by Naive Bayes and Natural Language Processing
</p>

</div>
""", unsafe_allow_html=True)

# ---------- Metrics ----------

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="metric">
    <h2>⚡ Fast</h2>
    <p>Instant Prediction</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric">
    <h2>🧠 AI Model</h2>
    <p>Naive Bayes</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric">
    <h2>📚 NLP</h2>
    <p>Count Vectorizer</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------- Main Card ----------

st.markdown('<div class="card">', unsafe_allow_html=True)

message = st.text_area(
    "✍️ Enter Message",
    height=200,
    placeholder="Example: Congratulations! You won a free gift..."
)

c1, c2 = st.columns(2)

with c1:
    predict = st.button("🚀 Predict")

with c2:
    clear = st.button("🗑️ Clear")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Prediction ----------

if predict:

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:

        data = vectorizer.transform([message])

        prediction = model.predict(data)
        prob = model.predict_proba(data)

        ham_prob = round(prob[0][0] * 100, 2)
        spam_prob = round(prob[0][1] * 100, 2)

        st.write("")
        st.subheader("📊 Prediction Result")

        if prediction[0] == 1:

            st.error("❌ SPAM MESSAGE DETECTED")

            st.progress(int(spam_prob))

            st.metric(
                "Spam Confidence",
                f"{spam_prob}%"
            )

        else:

            st.success("✅ SAFE MESSAGE")

            st.progress(int(ham_prob))

            st.metric(
                "Ham Confidence",
                f"{ham_prob}%"
            )

        st.write("")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Ham Probability",
                f"{ham_prob}%"
            )

        with c2:
            st.metric(
                "Spam Probability",
                f"{spam_prob}%"
            )

        st.balloons()