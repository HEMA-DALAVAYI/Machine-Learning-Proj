# Machine-Learning-Proj
The goal of this project is to build machine learning models for automated classification of academic research papers using ensemble methods.
# Overview
This project explores machine learning techniques to automatically classify research papers based on metadata (e.g., title, abstract). We compare Random Forest, Logistic Regression, and Recurrent Neural Networks (RNN), and implement an ensemble model using the two best-performing classifiers to improve overall accuracy and robustness.
# Requirements

- Python 3.7+
- Libraries:
  - `scikit-learn`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`
  - `nltk`
  - `tensorflow`
  - `keras-tuner`

If any of the libraries are not pre-installed, install them using the command `pip install <libraryName>`

## Dataset Description

The dataset used in this project is the **[Arxiv Scientific Research Papers Dataset](https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset)**, which consists of metadata from scientific papers published on arXiv.

### Fields Included

- **`id`**: Unique identifier for the paper  
- **`title`**: Title of the research paper  
- **`abstract`**: Abstract or summary of the paper  
- **`categories`**: Subject categories 
- **`update_date`**: Last updated timestamp for the record
- many other fields

### Fields Used for Classification

For the purpose of this machine learning project, we primarily use the following fields:

- **`title`**
- **`abstract`**
- Label --> **`categories`**

These features are used to train models that automatically classify research papers into appropriate academic categories.

## Steps to Run Each Model

After downloading the dataset, follow these steps to train, evaluate, and ensemble the models.  
**Note:** Each model script includes its own preprocessing steps, so there is no need to run a separate preprocessing script.

### 1. Train Random Forest Models

This script trains a **Random Forest classifier** for scientific paper classification based on preprocessed title and summary text using **TF-IDF vectorization**.

## 📂 Dataset Preparation

Ensure your dataset CSV file (e.g., `dataset.csv`) is placed in the same directory as `RandomForestModel.py`.

The CSV should have the following columns:
- `title`: Title of the paper
- `summary`: Abstract of the paper
- `category`: Ground-truth label

## ⚙️ Update the Dataset Path

Inside `RandomForestModel.py`, update this line if needed:

```python
X_raw, y = load_and_process_dataset('./dataset.csv')  # <-- Replace with your file path if different
```

## 🚀 Run the Script

Open your terminal, navigate to the project directory, and run:

```bash
python RandomForestModel.py
```

## 🔧 Optional Configuration

To change TF-IDF features, edit this line:

```python
X, vectorizer = tfidf_vectorize(X_raw, max_features=5000)  # You can adjust max_features
```

To tune Random Forest parameters, update the `random_forest()` function:

```python
RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
```

## 📊 Output

After training, the script will automatically evaluate the model and display:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

These metrics reflect the model’s performance in classifying papers into the correct category.


### 2. Train Logistic Regression Forest
Ensure that the dataset file is in the **same directory** as `LogisticRegModel.py`.

# Update the Dataset Path

Open the `LogisticRegModel.py` script and locate the line where the dataset path is defined.  
Update it to match your dataset filename:

    dataset_path = "your_dataset.csv"

### Run the Script from Terminal

Open your terminal, navigate to the project directory, and run:

    python LogisticRegModel.py

### Tune Hyperparameters (Optional)

Inside `LogisticRegModel.py`, you can modify parameters to adjust training behavior:

    learning_rate = 0.01
    num_epochs = 1000

### View Evaluation Results

Once training is complete, the script will **automatically evaluate the model**  
and display performance metrics such as **accuracy** , **loss**  and the **confusion matrix** in the terminal.

### 3. SimpleRNN

Trains a Simple RNN model.

```bash
python simple_rnn.py
```

### 3. BiLSTM

Trains a Bidirectional LSTM model.

```bash
python BiLSTMModel.py
```

BiLSTM uses NLTK for preprocessing and TensorFlow/Keras for modeling. Tune:
- `MAX_LEN`, `EMBEDDING_DIM`
- `dropout`, `num_units`, etc.

### 4. Ensemble (LogReg + BiLSTM)

Combines Logistic Regression and BiLSTM using soft or hard voting.

```bash
python EnsembleModel.py
```

Soft Voting:
```python
ensemble_predict_soft(log_model, bilstm_model, tokenizer, X_text, X_df)
```

Hard Voting:
```python
ensemble_predict_hard(log_model, bilstm_model, tokenizer, X_text, X_df, threshold=0.4)
```

## Output for SimpleRNN, BiLSTM, and Ensemble Model

Each script will report:
- Precision
- Recall
- F1-score
- Confusion Matrix

These metrics evaluate how well the model classifies research paper categories.




