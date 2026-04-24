# SMS spam classification
This project explores spam detection using both classical ML methods and transformer-based model.
## Problem
The goal is to classify SMS messages as **spam** or **ham** (not spam). This is a binary classification of texts problem with class imbalance.
## Dataset
We are going to use [SMSSpamCollection dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/code)
-5572 messages
-Class imbalance (13% spam messages)
Such imbalance makes recall, f1 metrics more valueable.
## Models
For this task two models were used
### 1. Baseline model
- TF-IDF vectorization
- Logistic Regression
### 2. Transformer model
- Pretrained bert-base-uncased
- Fine tuning for binary classification.
## Results

| Model                 | Precision | Recall | F1-score |
|-----------------------|----------|--------|----------|
| TF-IDF + Logistic Reg | 1.000    | 0.799  | 0.888    |
| BERT-base             | 0.960    | 0.966  | 0.963    |
