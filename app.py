
import streamlit as st
import joblib
from pathlib import Path

# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="centered"
)

# ---------------------------------
# File Paths
# ---------------------------------

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "sentiment_analysis_model.pkl"
EMOTION_PATH = BASE_DIR / "Emotions.pkl"


# ---------------------------------
# Load Model and Emotions
# ---------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_emotions():
    return joblib.load(EMOTION_PATH)


# ---------------------------------
# Custom CSS Design
# ---------------------------------

st.markdown("""
<style>

/* Main background */

.stApp {
    background: linear-gradient(
        135deg,
        #0f1028,
        #171638,
        #20204b
    );
    color: #ffffff;
}

/* Main content width */

.block-container {
    padding-top: 3rem;
    padding-bottom: 3rem;
    max-width: 850px;
}

/* Main heading */

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    background: linear-gradient(
        90deg,
        #a78bfa,
        #60a5fa,
        #22d3ee
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 5px;
}

/* Subtitle */

.subtitle {
    text-align: center;
    color: #c4c5e0;
    font-size: 16px;
    margin-bottom: 30px;
}

/* Input area */

.stTextArea textarea {
    background-color: #25264a !important;
    color: white !important;
    border: 1px solid #55558c !important;
    border-radius: 12px !important;
    font-size: 16px !important;
}

.stTextArea textarea:focus {
    border: 1px solid #a78bfa !important;
    box-shadow: 0 0 8px #8b5cf6 !important;
}

/* Predict button */

.stButton button {
    background: linear-gradient(
        90deg,
        #7c3aed,
        #2563eb
    );
    color: white;
    border: none;
    border-radius: 12px;
    height: 50px;
    font-size: 17px;
    font-weight: 700;
    transition: 0.3s;
}

.stButton button:hover {
    background: linear-gradient(
        90deg,
        #9333ea,
        #3b82f6
    );
    border: none;
    box-shadow: 0 0 15px #7c3aed;
    transform: translateY(-2px);
}

/* Result card */

.result-card {
    background: linear-gradient(
        135deg,
        #292957,
        #30306b
    );
    border: 1px solid #6666a5;
    border-radius: 16px;
    padding: 25px;
    margin-top: 15px;
    text-align: center;
}

.result-label {
    color: #c4b5fd;
    font-size: 15px;
    font-weight: 600;
}

.result-emotion {
    color: #ffffff;
    font-size: 32px;
    font-weight: 800;
    margin-top: 8px;
}

/* Section headings */

.section-heading {
    color: #c4b5fd;
    font-size: 21px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* Footer */

.footer {
    text-align: center;
    color: #999ac4;
    font-size: 13px;
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------
# Application Header
# ---------------------------------

st.markdown(
    '<div class="main-title">💬 Sentiment Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Understand emotions hidden behind words using Machine Learning.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ---------------------------------
# Load Application Resources
# ---------------------------------

try:

    model = load_model()
    Emotions = load_emotions()

    # Convert model labels into emotion names
    def get_emotion(label):

        if isinstance(Emotions, dict):
            return Emotions.get(
                label,
                Emotions.get(str(label), str(label))
            )

        try:
            return Emotions[int(label)]

        except (ValueError, TypeError, IndexError):
            return str(label)


    # ---------------------------------
    # Input Section
    # ---------------------------------

    st.markdown(
        '<div class="section-heading">✍️ Enter Your Text</div>',
        unsafe_allow_html=True
    )

    text_input = st.text_area(
        "Write something to analyse:",
        placeholder="Example: This product is amazing! I absolutely love it.",
        height=150,
        label_visibility="collapsed"
    )

    st.caption("💡 Enter a sentence, product review, or any text.")


    # ---------------------------------
    # Prediction Button
    # ---------------------------------

    predict = st.button(
        "🔍 Analyse Sentiment",
        use_container_width=True
    )


    # ---------------------------------
    # Prediction Section
    # ---------------------------------

    if predict:

        if not text_input.strip():

            st.warning("Please enter some text before predicting.")

        else:

            prediction = model.predict([text_input])[0]

            resulting_emotion = get_emotion(prediction)

            st.markdown(
                '<div class="section-heading">📊 Analysis Result</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        PREDICTED EMOTION
                    </div>
                    <div class="result-emotion">
                        {resulting_emotion}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


            # ---------------------------------
            # Emotion Probabilities
            # ---------------------------------

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    [text_input]
                )[0]

                classes = model.classes_

                confidence = max(probabilities) * 100

                st.markdown(
                    '<div class="section-heading">'
                    '🎯 Emotion Probabilities'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.metric(
                    label="Prediction Confidence",
                    value=f"{confidence:.2f}%"
                )

                # Sort emotions by probability
                emotion_results = []

                for label, probability in zip(
                    classes,
                    probabilities
                ):

                    emotion = get_emotion(label)

                    emotion_results.append(
                        (emotion, probability)
                    )

                emotion_results.sort(
                    key=lambda x: x[1],
                    reverse=True
                )

                with st.container(border=True):

                    for emotion, probability in emotion_results:

                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.write(f"**{emotion}**")

                        with col2:
                            st.write(
                                f"**{probability:.2%}**"
                            )

                        st.progress(float(probability))


    # ---------------------------------
    # Footer
    # ---------------------------------

    st.markdown(
        """
        <div class="footer">
            Built with ❤️ using Streamlit<br>
            Powered by CountVectorizer and Logistic Regression
        </div>
        """,
        unsafe_allow_html=True
    )


except FileNotFoundError as e:

    st.error(
        f"Required file not found: {e.filename}. "
        "Please check that your model and emotion files "
        "are in the application folder."
    )


except Exception as e:

    st.error(f"An error occurred: {e}")

