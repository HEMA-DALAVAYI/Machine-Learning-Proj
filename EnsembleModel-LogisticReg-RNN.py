# ensemble_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from kerastuner.tuners import RandomSearch

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("GPU available. Using GPU.")
    except:
        print("Could not set GPU. Using CPU instead.")
else:
    print("GPU not found. Running on CPU.")

# 1. Load and preprocess dataset
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
    return df[['title', 'summary', 'text', 'categories']]


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# 2. Train BiRNN Model
def train_birnn(X_train, y_train, vocab_size=30000, max_len=500):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)

    def build_model(hp):
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=hp.Int('embedding_dim', 100, 200, step=50)))
        model.add(Bidirectional(LSTM(hp.Int('lstm_units', 64, 256, step=64), return_sequences=False)))
        model.add(Dropout(hp.Float('dropout_rate', 0.3, 0.6, step=0.1)))
        model.add(Dense(hp.Int('dense_units', 64, 128, step=32), activation='relu'))
        model.add(Dense(y_train.shape[1], activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=2,
        executions_per_trial=1,
        directory='tuner_results',
        project_name='ensemble_birnn'
    )

    tuner.search(X_train_pad, y_train, epochs=5, validation_split=0.2, batch_size=64,
                 callbacks=[EarlyStopping(patience=2)])

    model = tuner.get_best_models(num_models=1)[0]
    return model, tokenizer

# 3. Train Logistic Regression
def train_logistic(X_train):
    vectorizer = ColumnTransformer([
        ('title_tfidf', TfidfVectorizer(max_features=5000, stop_words='english'), 'title'),
        ('summary_tfidf', TfidfVectorizer(max_features=10000, stop_words='english'), 'summary')
    ])
    model = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    return model

# 4. Soft Voting Prediction
def ensemble_predict_soft(log_model, birnn_model, tokenizer, X_text, X_df, max_len=500):
    log_proba = log_model.predict_proba(X_df)
    birnn_input = pad_sequences(tokenizer.texts_to_sequences(X_text), maxlen=max_len)
    birnn_proba = birnn_model.predict(birnn_input)

    if isinstance(log_proba, list):
        log_proba = np.array([np.vstack(p).T for p in zip(*log_proba)]).squeeze()
    if birnn_proba.shape != log_proba.shape:
        birnn_proba = birnn_proba[:, :log_proba.shape[1]]

    return (log_proba + birnn_proba) / 2

# 5. Hard Voting Prediction
def ensemble_predict_hard(log_model, birnn_model, tokenizer, X_text, X_df, threshold=0.4, max_len=500):
    log_pred = log_model.predict(X_df)
    birnn_input = pad_sequences(tokenizer.texts_to_sequences(X_text), maxlen=max_len)
    birnn_pred = (birnn_model.predict(birnn_input) > threshold).astype(int)

    if birnn_pred.ndim == 1:
        birnn_pred = birnn_pred.reshape(-1, 1)

    combined = np.stack([log_pred, birnn_pred.argmax(axis=1)], axis=-1)
    hard_votes = [np.bincount(row).argmax() for row in combined]

    final_pred = np.zeros_like(birnn_pred)
    for i, label in enumerate(hard_votes):
        final_pred[i, label] = 1

    return final_pred

# 6. Main function
def main():
    filepath = 'dataset.csv'
    df = load_data(filepath)
    df['text'] = df['text'].apply(preprocess_text)

    X = df['text']
    X_meta = df[['title', 'summary']]

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['categories'])

    X_train_text, X_test_text, X_train_meta, X_test_meta, y_train, y_test = train_test_split(
        X, X_meta, y, test_size=0.2, random_state=42)

    log_model = train_logistic(X_train_meta)
    log_model.fit(X_train_meta, y_train.argmax(axis=1))

    birnn_model, tokenizer = train_birnn(X_train_text, y_train)

    y_pred_proba = ensemble_predict_soft(log_model, birnn_model, tokenizer, X_test_text, X_test_meta)
    y_pred_soft = (y_pred_proba > 0.4).astype(int)

    print("\n--- Soft Voting Results ---")
    print(classification_report(y_test, y_pred_soft, target_names=mlb.classes_))

    y_pred_hard = ensemble_predict_hard(log_model, birnn_model, tokenizer, X_test_text, X_test_meta)

    print("\n--- Hard Voting Results ---")
    print(classification_report(y_test, y_pred_hard, target_names=mlb.classes_))

if __name__ == '__main__':
    main()
