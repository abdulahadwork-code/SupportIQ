import os
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_metrics(eval_pred):

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, predictions, average='macro')
    return {"macro_f1": macro_f1}

def train_distilbert():

    df = pd.read_csv("ml/data/cleaned_tickets.csv")
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['category'])
    num_classes = len(label_encoder.classes_)
    import joblib
    joblib.dump(label_encoder, "ml/models/label_encoder.pkl")
    train_df, test_df = train_test_split(
        df[['text', 'label']], test_size=0.2, random_state=42, stratify=df['label']
    )

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    logging.info("Tokenizing data with DistilBERT tokenizer")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=128)
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    logging.info("Loading DistilBERT model for sequence classification")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", 
        num_labels=num_classes
    )

    training_args = TrainingArguments(
        output_dir="ml/models/distilbert_checkpoints",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",          
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to="none"                
    )
        
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,      
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logging.info("Start fine tuning")
    trainer.train()
    logging.info("Evaluating")
    eval_results = trainer.evaluate()
    logging.info(f"DistilBERT Evaluation Results: {eval_results}")
    predictions = trainer.predict(tokenized_test)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    true_labels = predictions.label_ids
    pred_classes = label_encoder.inverse_transform(pred_labels)
    true_classes = label_encoder.inverse_transform(true_labels)
    
    logging.info("="*50)
    logging.info("DISTILBERT PER CLASS PERFORMANCE")
    logging.info("="*50)
    print(classification_report(true_classes, pred_classes, target_names=label_encoder.classes_))
    final_model_path = "ml/models/distilbert_ticket_classifier"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    logging.info(f"DistilBERT model saved to {final_model_path}")

if __name__ == "__main__":
    train_distilbert()