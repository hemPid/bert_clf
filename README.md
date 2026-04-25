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

## Key Findings

- Baseline model already shows good results due to clear lexical patterns in spam messages.

- BERT improves recall by capturing contextual information.

- However, overall gains are moderate due to the simplicity of the dataset.

## Error Analysis
Baseline shows good performance due to clear lexical patterns in spam messages. But it can miss spam messages without explicit keywords.

For example:

"TheMob>Hit the link to get a premium Pink Panther game, the new no. 1 from Sugababes, a crazy Zebra animation or a badass Hoody wallpaper-all 4 FREE!"

The baseline model classifies it as ham, while BERT correctly identifies it as spam. BERT performs better on such cases due to contextual understanding.
