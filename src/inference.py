import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os


def baseline_predict(text, return_str=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'models', 'tfidf_logreg_baseline.joblib')
    if isinstance(text, str):
        text = [text]
    model = joblib.load(model_path)
    preds = model.predict(text)
    if not return_str:
        return preds
    results = []
    for i in range(len(text)):
        label = 'spam' if preds[i] == 1 else 'ham'
        results.append(label)

    if len(results) == 1:
        return results[0]
    return results



def bert_predict(text, return_str=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'models', 'bert')
    if isinstance(text, str):
        text = [text]
    
    tokenizer = BertTokenizer.from_pretrained(model_path+'\\sms_spam_tokenizer\\')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load(model_path+'\\best_model.pt', map_location=device))
    model.to(device)
    model.eval()
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_class_ids = torch.argmax(probs, dim=-1)
    
    results = []
    for i in range(len(text)):
        if return_str:
            label = 'spam' if pred_class_ids[i].item() == 1 else 'ham'
        else:
            label = pred_class_ids[i].item()
        results.append(label)

    if len(results) == 1:
        return results[0]
    return results





if __name__ == '__main__':
    txt = "only today take two tickets to ping pong show by 200 dollars. Call on 9683580632"
    print(baseline_predict(txt))
    print(bert_predict(txt))