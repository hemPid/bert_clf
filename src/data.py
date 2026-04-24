import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path='../data/SMSSpamCollection'):
    df = pd.read_csv(path, sep='\t', header=None, names=['target', 'text'], encoding='utf-8')
    df['target'] = df['target'].map({'ham': 0, 'spam': 1})
    return df


def get_train_test_splits(test_size=0.2):
    df = load_data()
    return train_test_split(
        df['text'].tolist(),
        df['target'].tolist(),
        test_size=test_size,
        random_state=42,
        stratify=df['target']
    )
