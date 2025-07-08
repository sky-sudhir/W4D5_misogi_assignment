from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
import os

MODEL_PATH = "./model"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = "./sanitized_output.csv"

app = FastAPI()

class SalesDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class PredictRequest(BaseModel):
    transcript: str

@app.get("/health")
def health():
    return {"status": "API is running"}

@app.post("/train")
def train_model():
    df = pd.read_csv(DATA_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    X_train, X_test, y_train, y_test = train_test_split(df["transcript"], df["label"], test_size=0.2, random_state=42)
    train_dataset = SalesDataset(X_train.tolist(), y_train.tolist(), tokenizer)
    eval_dataset = SalesDataset(X_test.tolist(), y_test.tolist(), tokenizer)

    def compute_metrics(p):
        preds = torch.argmax(torch.tensor(p.predictions), dim=1)
        labels = torch.tensor(p.label_ids)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    return {"message": "Model trained and saved."}

@app.post("/predict")
def predict(request: PredictRequest):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    inputs = tokenizer(request.transcript, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs).item()
        confidence = probs[0][label].item()

    return {
        "prediction": "success" if label == 1 else "failure",
        "confidence": round(confidence * 100, 2)
    }
