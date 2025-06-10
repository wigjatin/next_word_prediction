import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🔮",
    layout="centered"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 800px;
    }
    h1 {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem !important;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #e1e5e9;
        padding: 12px 16px;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .stInfo {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border: none;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        margin: 8px 0;
    }
    .stInfo:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
    }
    h3 {
        color: #333;
        border-bottom: 3px solid #667eea;
        padding-bottom: 8px;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    code {
        padding: 20px !important;
        border-radius: 12px !important;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
        border: 1px solid #dee2e6 !important;
    }
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .stWarning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        border: none;
        border-radius: 12px;
        padding: 16px;
    }
    .stMarkdown p {
        background: white;
        padding: 12px 16px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stMarkdown p:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    .stInfo, .stMarkdown p {
        animation: fadeInUp 0.5s ease-out;
    }
    @media (max-width: 768px) {
        h1 {
            font-size: 2rem !important;
        }
        .main .block-container {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1>🔮 Next Word Predictor</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter some words and discover what AI thinks comes next!</div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #666; margin-bottom: 2rem;">
Enter at least 3 words to see predictions for the next word.<br>
The AI model was trained on text data using LSTM neural networks.
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    model = load_model('next_words.h5')
    tokenizer = pickle.load(open('token.pkl', 'rb'))
    return model, tokenizer

model, tokenizer = load_resources()

def predict_top_words(model, tokenizer, text, top_n=5):
    words = text.strip().split()
    if len(words) < 3:
        return []
    input_sequence = ' '.join(words[-3:])
    sequence = tokenizer.texts_to_sequences([input_sequence])
    sequence = np.array(sequence)
    if sequence.size == 0:
        return []
    preds = model.predict(sequence, verbose=0)[0]
    top_indices = np.argsort(preds)[-top_n:][::-1]
    predicted_words = []
    for idx in top_indices:
        for word, index in tokenizer.word_index.items():
            if index == idx:
                predicted_words.append(word)
                break
    return predicted_words

st.sidebar.markdown("### 🔗 Source Code")
st.sidebar.markdown("[View on GitHub](https://github.com/wigjatin/next_word_prediction)")
st.sidebar.markdown("---")
st.sidebar.markdown("### How it works:")
st.sidebar.markdown("""
1. Enter at least 3 words
2. AI analyzes the last 3 words
3. Predicts the most probable next words
4. Shows both predicted words and full sentences
""")

user_input = st.text_input("Enter your text here:", "I am going")
predict_button = st.button("Predict Next Words")

if predict_button or user_input:
    if user_input:
        predictions = predict_top_words(model, tokenizer, user_input)
        if predictions:
            full_sentences = [f"{user_input} {word}" for word in predictions]
            st.subheader("Top Predicted Words:")
            cols = st.columns(len(predictions))
            for i, word in enumerate(predictions):
                with cols[i]:
                    st.info(f"**{word}**")
            st.subheader("Complete Sentence Predictions:")
            for i, sentence in enumerate(full_sentences, 1):
                st.markdown(f"{i}. **{sentence}**")
            st.markdown("---")
        else:
            st.warning("Could not generate predictions. Please enter at least 3 valid words.")
    else:
        st.warning("Please enter some text")