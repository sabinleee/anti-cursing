import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = "C:/Users/withus/Desktop/sabin/oss/"
train = pd.read_csv(PATH+"Dataset/unsmile_train_v1.0.tsv", sep="\t")
valid = pd.read_csv(PATH+"Dataset/unsmile_valid_v1.0.tsv", sep="\t")

class myDataset(torch.utils.data.Dataset):
    def __init__(self, encoding, label):
        self.encoding = encoding
        self.label = label
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encoding.items()}
        item["label"] = torch.tensor(self.label[idx])
        return item

    def __len__(self):
        return len(self.label)


def sentence_predict(sent, model, tokenizer):
    model.eval()

    tokenized_sent = tokenizer(
        sent,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
        add_special_tokens=True,
    )

    tokenized_sent.to(device)

    with torch.no_grad():
        output = model(**tokenized_sent)

    logits = output[0].detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)

    if result == 0:
        return "악성댓글"
    else:
        return "정상댓글"

def compute_metrics(preds):
    labels = preds.label_ids
    preds = preds.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "acc": acc, 
        "precision": precision, 
        "recall": recall,
        "f1": f1,
    }

def train():
    MODEL_NAME = "beomi/KcELECTRA-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_train = tokenizer(
                        list(train["문장"]), 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=512, 
                        add_special_tokens=True
                    )
    tokenized_valid = tokenizer(
                        list(valid["문장"]), 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=512, 
                        add_special_tokens=True
                    )

    train_label = train['clean'].values
    valid_label = valid['clean'].values

    train_dataset = myDataset(tokenized_train, train_label)
    valid_dataset = myDataset(tokenized_valid, valid_label)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    training_args = TrainingArguments(
        output_dir="C:/Users/withus/Desktop/sabin/oss/output_binary",
        num_train_epochs=10,
        per_device_eval_batch_size=64,
        per_device_train_batch_size=8,
        logging_dir="C:/Users/withus/Desktop/sabin/oss/log_binary",
        save_steps=1000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.evaluate(eval_dataset=valid_dataset)
    
    return model, tokenizer


def main(model, tokenizer):
    while True:
        sent = input("문장을 입력하세요: ")
        if sent == "0":
            break
        print(sentence_predict(sent, model, tokenizer))
        print("-"*50)

if __name__ == "__main__":
    model, tokenizer = train()
    main(model, tokenizer)
    