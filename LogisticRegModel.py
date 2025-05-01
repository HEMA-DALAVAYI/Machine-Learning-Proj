import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(csv_path):
    # Load dataset
    df = pd.read_csv(csv_path)

    # Select feature columns (customize if needed)
    feature_columns = ['title', 'summary']
    label_column = 'category'

    # Features and labels
    x = df[feature_columns]
    y = df[label_column]

    # Train-test split (80-20)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2)

    return x_train, x_test, y_train, y_test


def logistic_reg(x_train, y_train):
    vectorizer = ColumnTransformer(
        transformers=[
            ('title_tfidf', TfidfVectorizer(max_features=5000, stop_words='english'), 'title'),
            ('summary_tfidf', TfidfVectorizer(max_features=10000, stop_words='english'), 'summary')
        ]
    )

    model = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    model.fit(x_train, y_train)

    return model




def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    
    precision = precision_score(y_test, y_pred, average='weighted')
    print(f"Precision: {precision}")
    
    recall = recall_score(y_test, y_pred, average='weighted')
    print(f"Recall: {recall}")
    
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1 Score: {f1}")
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(cm.shape[1]), yticklabels=np.arange(cm.shape[0]))
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()


def main():
    x_train, x_test, y_train, y_test = preprocess_data(r'C:\Users\siris\Downloads\mldata\\arXiv_scientific dataset.csv')
    model = logistic_reg(x_train, y_train)
    evaluate(model, x_test, y_test)