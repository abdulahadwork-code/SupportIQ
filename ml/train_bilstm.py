import os
import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, padding_idx=0):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes) 

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, cell) = self.bilstm(embedded)
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_concat = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        out = self.dropout(hidden_concat)
        return self.fc(out)

class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_bilstm():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    df = pd.read_csv("ml/data/cleaned_tickets.csv")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['category'])
    num_classes = len(label_encoder.classes_)
    import joblib
    joblib.dump(label_encoder, "ml/models/label_encoder.pkl")
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], y, test_size=0.2, random_state=42, stratify=y
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    train_dataset = TicketDataset(X_train.values, y_train, tokenizer)
    test_dataset = TicketDataset(X_test.values, y_test, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    model = BiLSTMClassifier(
        vocab_size=vocab_size, 
        embed_dim=128, 
        hidden_dim=128, 
        num_classes=num_classes
    ).to(device)

    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (num_classes * class_counts)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    epochs = 5
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        logging.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)         
            outputs = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)   
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    pred_labels = label_encoder.inverse_transform(all_preds)
    true_labels = label_encoder.inverse_transform(all_labels)
    
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    logging.info("="*50)
    logging.info(f"BiLSTM MODEL EVALUATION | Macro-F1: {macro_f1:.4f}")
    logging.info("="*50)
    print("\nPer-Class Performance:")
    print(classification_report(true_labels, pred_labels))
    os.makedirs("ml/models", exist_ok=True)
    torch.save(model.state_dict(), "ml/models/bilstm_classifier.pth")

if __name__ == "__main__":
    train_bilstm()