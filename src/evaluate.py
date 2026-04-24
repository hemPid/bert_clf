import numpy as np
import inference
import data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import torch


def evaluate_baseline():
    X_train, X_test, y_train, y_test = data.get_train_test_splits()
    y_pred = inference.baseline_predict(X_test, return_str=False)
    print('TF-IDF + LogReg baseline results on test:')
    print('Accuracy:', precision_score(y_test,y_pred))
    print('Precision:', precision_score(y_test,y_pred))
    print('Recall:', recall_score(y_test,y_pred))
    print('F1:', f1_score(y_test,y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))


def evaluate_bert():
    X_train, X_test, y_train, y_test = data.get_train_test_splits()
    y_pred = inference.bert_predict(X_test, return_str=False)
    print('BERT results on test:')
    print('Accuracy:', precision_score(y_test,y_pred))
    print('Precision:', precision_score(y_test,y_pred))
    print('Recall:', recall_score(y_test,y_pred))
    print('F1:', f1_score(y_test,y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))




if __name__=='__main__':
    evaluate_baseline()
    evaluate_bert()