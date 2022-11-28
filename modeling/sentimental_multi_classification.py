import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer , AutoModel
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn import metrics

class MultiLabelDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe["문장"]
        self.targets = self.data[cols].values
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.targets[index], dtype=torch.float)
        }
        
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = AutoModelForSequenceClassification.from_pretrained("beomi/KcELECTRA-base",num_labels=10)
        

    def forward(self, input_ids, attention_mask,token_type_ids):
        output = self.l1(input_ids=input_ids, attention_mask=attention_mask,token_type_ids=token_type_ids)
        return output



def compute_metrics(preds):
    labels = preds.label_ids
    preds = preds.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {
        "acc": acc, 
        "precision": precision, 
        "recall": recall,
        "f1": f1,
    }
    
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)
    
def train(epoch, training_loader, model, optimizer, device):
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0)):
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['labels'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs['logits'], targets)
        if _%5000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
def validation(testing_loader, model, device):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0)):
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['labels'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs["logits"]).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
    
def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

def sentence_classification(sentence, model, tokenizer, device):
    
    tokenized_sent = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        add_special_tokens=True,
    )
    
    tokenized_sent.to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(**tokenized_sent)
        
    outputs = output['logits'].detach().cpu().numpy()
    final_outputs = np.array(outputs) >=0.5
    
    return (pd.DataFrame(final_outputs,columns=cols))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data = pd.read_csv("./Dataset/unsmile_train_v1.0.tsv", sep="\t")
valid_data = pd.read_csv("./Dataset/unsmile_valid_v1.0.tsv", sep="\t")

cols = train_data.columns.tolist()
cols.remove("문장")
cols.remove("clean")

MODEL_NAME = "beomi/KcELECTRA-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 128
tokenized_train = tokenizer(
                    list(train_data["문장"]), 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=MAX_LEN, 
                    add_special_tokens=True
                )
tokenized_valid = tokenizer(
                    list(valid_data["문장"]), 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=MAX_LEN, 
                    add_special_tokens=True
                )
training_set = MultiLabelDataset(train_data, tokenizer, MAX_LEN)
testing_set = MultiLabelDataset(valid_data, tokenizer, MAX_LEN)
train_params = {'batch_size': 8,
                'shuffle': True,
                'num_workers': 0
                }
test_params = {'batch_size': 16,
                'shuffle': True,
                'num_workers': 0
                }
training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
model = BERTClass()
model.to(device)
training_args = TrainingArguments(
    output_dir="./output_1",
    num_train_epochs=10,
    per_device_eval_batch_size=16,
    per_device_train_batch_size=8,
    logging_dir="./log",
    save_steps=1000,
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=training_set,
    eval_dataset=testing_set,
    
)
LEARNING_RATE = 5e-5
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

def main():
    for epoch in range(5):
        train(epoch, training_loader, model, optimizer, device)
    # trainer.train()
    outputs, targets = validation(testing_loader, model, device)
    final_outputs = np.array(outputs) >=0.5
    val_hamming_loss = metrics.hamming_loss(targets, final_outputs)
    val_hamming_score = hamming_score(np.array(targets), np.array(final_outputs))
    print(f"Hamming Score = {val_hamming_score}")
    print(f"Hamming Loss = {val_hamming_loss}")


if __name__=="__main__":
    main()
    sentence_classification("그만하쟈", model, tokenizer, device)