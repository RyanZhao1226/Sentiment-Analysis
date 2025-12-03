# Twitter Sentiment Analysis

Dataset: [https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis]()

This project compares traditional machine learning models and a transformer-based model (DistilBERT) for sentiment analysis on Twitter data.

## Models

Traditional Models:

- Naive Bayes
- Logistic Regression
- AdaBoost
- XGBoost

Deep Learning Models:

- DistilBERT (best performance)

## Dataset & Processing

- Twitter sentiment dataset
- Binary labels: Positive / Negative
- Text cleaning, lemmatization, stopword removal
- TF–IDF for ML models; tokenization for DistilBERT

## Results

| Model               | F1 Score         |
| ------------------- | ---------------- |
| DistilBERT          | **0.9520** |
| Logistic Regression | 0.8928           |
| Naive Bayes         | 0.8457           |
| AdaBoost            | 0.8292           |
| XGBoost             | 0.8144           |

## Key Points

- DistilBERT achieves highest accuracy
- Logistic Regression offers best speed–performance balance
- XGBoost yields highest precision among ML models

## Future Work

- Hyperparameter tuning
- Model ensembles
- Error analysis
- Real-time API deployment
