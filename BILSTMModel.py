# filepath: arxiv_rnn_classifier.py

import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from kerastuner.tuners import RandomSearch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# GPU setup
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU available. Using GPU.")
    except:
        print("Could not set GPU. Using CPU instead.")
else:
    print("GPU not found. Running on CPU.")


# Text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Load and filter dataset
def load_data(filepath):
    allowed_categories = {
        "Artificial Intelligence",
        "Machine Learning",
        "Machine Learning (Statistics)",
        "Computer Vision and Pattern Recognition",
        "Computation and Language (Natural Language Processing)"
    }
    df = pd.read_csv(filepath)
    df.dropna(subset=['title', 'summary', 'category'], inplace=True)
    df = df[df['category'].isin(allowed_categories)]
    df['text'] = df['title'] + ' ' + df['summary']
    df['categories'] = df['category'].apply(lambda x: [x])
    return df[['text', 'categories']]

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# Tokenize and pad text
def tokenize_and_pad(texts, max_len=500, vocab_size=30000):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded, tokenizer

# Build BiLSTM model
def build_model(hp):
    model = Sequential()
    model.add(Embedding(input_dim=30000, output_dim=hp.Int('embedding_dim', 100, 200, step=50)))
    model.add(Bidirectional(LSTM(hp.Int('lstm_units', 64, 256, step=64), return_sequences=False)))
    model.add(Dropout(hp.Float('dropout_rate', 0.3, 0.6, step=0.1)))
    model.add(Dense(hp.Int('dense_units', 64, 128, step=32), activation='relu'))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Visualize class distribution
def plot_category_distribution(df):
    cat_counts = df['categories'].explode().value_counts()
    cat_counts.plot(kind='bar', title='Category Distribution')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig("bilstm_dist.png")

# Main function
def main():
    data_path = 'dataset.csv'
    df = load_data(data_path)
    df['text'] = df['text'].apply(preprocess_text)

    plot_category_distribution(df)

    X = df['text'].values
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['categories'])

    X_padded, tokenizer = tokenize_and_pad(X)

    global num_classes
    num_classes = y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=3,
        executions_per_trial=1,
        directory='tuner_results',
        project_name='arxiv_bilstm'
    )

    tuner.search(X_train, y_train, epochs=10, validation_split=0.2, batch_size=64,
                 callbacks=[EarlyStopping(patience=2)])

    best_model = tuner.get_best_models(num_models=1)[0]
    y_pred = (best_model.predict(X_test) > 0.4).astype(int)  # lowered threshold to improve recall

    labels = list(range(len(mlb.classes_)))
    print(classification_report(y_test, y_pred, labels=labels, target_names=mlb.classes_))

if __name__ == '__main__':
    main()
