# Disaster Tweets Classification Project

This project aims to classify tweets as either disaster-related or non-disaster-related using Natural Language Processing (NLP) and Machine Learning (ML) techniques. By analyzing textual data from social media, such as Twitter, the model provides real-time insights that can enhance disaster management and response systems.

---

## Objectives

- **Business Problem**: Classify tweets into disaster-related or non-disaster-related categories to improve disaster response efficiency.
- **Dataset**: Kaggle dataset containing 7,613 training samples and 3,263 test samples.
- **Machine Learning Goal**: Binary classification leveraging NLP techniques and various ML models.

---

## Methodology

### 1. Dataset Description
- **Features**: Tweet text, location, and keyword.
- **Class Distribution**: Disaster-related tweets (43%) vs. Non-disaster-related tweets (57%).

### 2. Data Preprocessing
- Removed HTML tags, emojis, and URLs.
- Lowercased text, removed stopwords, and tokenized.
- Handled null values in `keyword` and `location` fields through imputation.

### 3. Exploratory Data Analysis (EDA)
- Visualized unigram, bigram, and trigram frequencies.
- Analyzed disaster vs. non-disaster topics using topic modeling.
- Addressed class imbalance through targeted metrics.

### 4. Model Development
- **Baseline Models**:
  - Logistic Regression with TF-IDF vectorization.
  - Random Forest with hyperparameter tuning.
- **Advanced Models**:
  - LSTM with Glove embeddings.
  - Fine-tuned BERT (`bert-base-uncased`).
  - Ensemble model combining LSTM and BERT for improved performance.

### 5. Model Evaluation
- Metrics:
  - **Precision**: Disaster (81%), Non-disaster (85%).
  - **Recall**: Disaster (80%), Non-disaster (86%).
  - **F1-Score**: Disaster (0.80), Non-disaster (0.85).
  - **ROC AUC**: 0.89 (best ensemble model).
- Confusion Matrices and ROC curves were plotted for each model to assess classification performance.

---

## Results and Key Metrics

### Model Performance Comparison

| Model               | Precision (Disaster) | Recall (Disaster) | F1-Score (Disaster) | ROC AUC |
|---------------------|-----------------------|-------------------|---------------------|---------|
| Logistic Regression | 0.86                 | 0.70              | 0.77                | 0.87    |
| Random Forest       | 0.81                 | 0.69              | 0.74                | 0.85    |
| LSTM (Base)         | 0.72                 | 0.70              | 0.71                | 0.84    |
| LSTM (Glove)        | 0.75                 | 0.73              | 0.75                | 0.86    |
| BERT                | 0.81                 | 0.80              | 0.80                | 0.89    |
| Ensemble (BERT + LSTM) | 0.93              | 0.91              | 0.92                | 0.92    |

### Highlights
- **Best Model**: Ensemble (BERT + LSTM) with an accuracy of 92% and F1-score of 0.92 for disaster-related tweets.
- **BERT Performance**: Achieved consistent results across all metrics with minimal false positives and false negatives.

---

## Tech Stack

- **Programming**: Python (Pandas, NumPy, TensorFlow, PyTorch, Scikit-learn)
- **NLP Techniques**: TF-IDF, Glove Embeddings, Transformer Models (BERT)
- **Visualization**: Matplotlib, Seaborn

---

## Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/Akshay-DisasterTweets/Disaster-Tweets.git
   cd Disaster-Tweets1 ```
   
---

## Future Work

Handle Class Imbalance: Implement advanced sampling techniques.
Attention Mechanisms: Incorporate self-attention layers in simpler models.
Multimodal Approaches: Combine textual and visual data for enhanced classification.
Expanded Preprocessing: Experiment with more preprocessing techniques like advanced N-grams.
