Next Word Prediction using LSTM
https://via.placeholder.com/800x400?text=Next+Word+Prediction+Demo
Live Demo: https://nextwordprediction-jatinwig.streamlit.app/

Overview
This project implements a next word prediction model using TensorFlow and LSTM neural networks. The model analyzes sequences of words to predict the most probable next word in a sentence, with applications in keyboard suggestions, autocompletion systems, and natural language processing workflows.

Key Features
LSTM-based neural network for sequence prediction

Tokenization and text preprocessing pipeline

Top-5 word predictions with probability scores

Streamlit-powered web interface

Easy-to-deploy model architecture

Live Demo
Experience the model in action:
https://nextwordprediction-jatinwig.streamlit.app/

Model Architecture
python
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=3))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation='relu'))
model.add(Dense(vocab_size, activation="softmax"))
Requirements
Python 3.8+

TensorFlow 2.x

NumPy

scikit-learn

Streamlit

Pickle

Installation
bash
# Clone the repository
git clone https://github.com/wigjatin/next_word_prediction.git
cd next_word_prediction

# Install dependencies
pip install -r requirements.txt
Usage
Training the Model:

python
python train.py
Running the Streamlit App:

bash
streamlit run app.py

Project Structure
text
next_word_prediction/
├── app.py                # Streamlit application
├── train.py              # Model training script
├── predict.py            # Prediction functions
├── next_words.h5         # Trained model weights
├── token.pkl             # Tokenizer object
├── data.txt              # Training dataset
├── requirements.txt      # Python dependencies
└── README.md
Training Parameters
Parameter	Value
Epochs	2
Batch Size	64
Learning Rate	0.001
Sequence Length	3 words
Vocabulary Size	11,000+
Embedding Dimension	10
LSTM Units	1000
How It Works
Text Preprocessing:

Load and clean text data

Tokenize words and create sequences

Model Training:

Create embedding layer for word representations

Stack LSTM layers to capture sequence patterns

Use dense layers for final predictions

Prediction:

Convert input text to token sequences

Use trained model to predict top 5 next words

Display results in user-friendly format

Contributing
Contributions are welcome! Please follow these steps:

Fork the repository

Create a new branch (git checkout -b feature/your-feature)

Commit your changes (git commit -am 'Add some feature')

Push to the branch (git push origin feature/your-feature)

Open a pull request

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For questions or feedback, please contact:

Jatin Wig - wigjatin2@gmail.com

Project Repository: https://github.com/wigjatin/next_word_prediction

Experience the live demo: https://nextwordprediction-jatinwig.streamlit.app/

