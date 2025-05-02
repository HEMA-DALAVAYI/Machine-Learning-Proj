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

### 3. Train BILSTM Model



### 3. Train Ensemble Model




