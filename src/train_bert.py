import torch
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import numpy as np
import data


class SMSDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['target'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)




def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc





def train():
    model_path = '../models/bert/'
    X_train, X_test, y_train, y_test = data.get_train_test_splits()
    print('Loading model')
    MAX_LEN = 128
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=MAX_LEN)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=MAX_LEN)
    
    train_dataset = SMSDataset(train_encodings, y_train)
    test_dataset = SMSDataset(test_encodings, y_test)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    epochs = 3
    best_val_acc = 0.0
    log_interval = 50
    print('Start training')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['target'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if (step + 1) % log_interval == 0:
                current_loss = loss.item()
                print(f"Epoch {epoch+1}, Step {step+1}/{len(train_loader)}, Loss: {current_loss:.4f}")

        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1} compl. | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path+'best_model.pt')
            tokenizer.save_pretrained(model_path+'sms_spam_tokenizer')
            print(f"  -> Model saved. (acc={val_acc:.4f})")


if __name__=='__main__':
    train()