# Next Word Prediction System with LSTM

A high-performance next word prediction model built using Natural Language Processing (NLP) techniques and deep learning. This project uses a custom-trained LSTM-based neural network to predict the most probable word following a given sequence of three words.

---

## Problem Statement

Typing assistants and writing tools benefit greatly from intelligent word prediction systems. From chat apps to email clients and content editors, predicting the next word boosts productivity and enhances user experience.

This project builds an effective word predictor to suggest the most likely next word based on context.

---

## Features

- Trained on a custom English text corpus
- Keras Tokenizer for text preprocessing
- Sequential LSTM architecture with dense layers
- Achieves highly relevant top-5 word predictions
- Pickled tokenizer and saved model for inference
- Streamlit-based live demo for interactive usage

---

## Model Selection

After evaluating several sequence modeling approaches, **LSTM (Long Short-Term Memory)** was selected due to its:

- Strong performance on sequence and context understanding
- Ability to learn temporal dependencies in language
- Scalability for longer training on larger corpora

---

## Performance

| Model          | Layers   | Optimizer | Notes                     |
|----------------|----------|-----------|----------------------------|
| LSTM (final)   | 2 LSTM + Dense | Adam     | Top-5 predictions highly relevant |
| Simple RNN     | 1 RNN + Dense | Adam     | Lower accuracy and context retention |
| GRU            | 1 GRU + Dense | Adam     | Competitive but LSTM had better results |

---

## Tech Stack

- Python 
- TensorFlow / Keras
- NumPy / Pickle
- Streamlit (for frontend demo)

---

## Text Preprocessing

- Cleaning whitespaces, quotes, newline characters
- Lowercasing
- Tokenization via Keras `Tokenizer`
- Creating sequences of 4 words: first 3 as input, last as label
- One-hot encoding of labels using `to_categorical`

---

## Demo
You can access the live demo of the application by visiting the following link:  
[View Demo](https://nextwordprediction-jatinwig.streamlit.app/)
