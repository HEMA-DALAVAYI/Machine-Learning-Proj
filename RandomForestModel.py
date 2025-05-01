import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Load and Prepare Dataset
def load_and_process_dataset(path):
    data = pd.read_csv(path) 
    print("Text Preprocessing")
    data['text'] = data.apply(lambda row: text_preprocessing(row['title'], row['summary']), axis=1)
    X_raw = data['text']
    y = data['category']
    return X_raw, y

# Text Preprocessing
def text_preprocessing(title, summary):
    combined_text = f"{title or ''} {summary or ''}".lower()  # Lowercase
    clean = re.sub(r'[^a-zA-Z\s]', '', combined_text)  # Remove special characters
    return clean

# TF-IDF Vectorization
def tfidf_vectorize(X_raw, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(X_raw)
    return X, vectorizer

# Build Random Forest Model
def random_forest():
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Train the Model
def fit_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Evaluate the Model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")



print("Load Dataset")
X_raw, y = load_and_process_dataset('./dataset.csv')

print("TF-IDF Vectorizing")
X, vectorizer = tfidf_vectorize(X_raw)

print("80:20 dataset splitting")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train model")
model = random_forest()
model = fit_model(model, X_train, y_train)

print("Evaluate model")
evaluate_model(model, X_test, y_test)
