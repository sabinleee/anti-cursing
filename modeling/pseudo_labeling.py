
# sentence_add_auto(valid.head(5),["한남충 씨발아",0,1,0,0,0,0,0,0,1,0,0])
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,random_split

from tqdm import tqdm

from transformers import BertModel,BertForSequenceClassification,AutoModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

pd.set_option('display.max_rows', None)
class PseudoDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_sentence):
        self.tokenized_sentence = tokenized_sentence
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.tokenized_sentence.items()}
        
        return item

    def __len__(self):
        return len(tokenized_psuedo["input_ids"])
    

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
    
    def get_classes(self):
        return self.label
    
    
def pseudo_data_eval(pseudo_loader):
    
    model.eval()
    
    pseudo_outputs=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(pseudo_loader)):
            
            ids = data['input_ids'].to(device, dtype = torch.long)
            mask = data['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            pseudo_outputs.extend(torch.sigmoid(outputs["logits"]).cpu().detach().numpy().tolist())
            
    
    return pseudo_outputs

def pseudo_labeled_data_generator(pseudo_outputs,dataset,k=0.997):
    
    pseudo_outputs = np.asarray(pseudo_outputs)
    boolean_index = (pseudo_outputs).max(axis=1) > k
    
    labels = pseudo_outputs[boolean_index].argmax(axis=1)
    
    pseudo_dataset = pd.DataFrame(columns=['문장','여성/가족','남성','성소수자','인종/국적','연령','지역','종교','기타 혐오','악플/욕설','clean','개인지칭'])
    pseudo_dataset["문장"] = dataset[boolean_index]
    pseudo_dataset['clean'] = labels
    pseudo_dataset.fillna(0,inplace=True)
    
    return pseudo_dataset

# def psuedo_labeled_data_add(psuedo_outputs,dataset,k=0.997):
    
#     psuedo_outputs = np.asarray(psuedo_outputs)
#     boolean_index = (psuedo_outputs).max(axis=1) > k

#     labels = psuedo_outputs[boolean_index].argmax(axis=1)

#     psuedo_dataset = dataset[boolean_index]
#     psuedo_dataset.loc[:,'clean'] = labels
    
#     return psuedo_dataset

# original_dataset: 원래 있던 dataset(train dataset), psuedo_sentences: unlabeled된 sentences
def pseudo_labeling(pseudo_sentences,original_dataset=None):
    
    unlabeled_dataset = PseudoDataset(tokenized_psuedo)
    pseudo_params = {'batch_size': 8,
                    'shuffle': False,
                    'num_workers': 0
                    }

    pseudo_loader = DataLoader(unlabeled_dataset, **pseudo_params)
    pseudo_outputs = pseudo_data_eval(pseudo_loader)
    pseudo_labeled_dataset = pseudo_labeled_data_generator(pseudo_outputs,pseudo_sentences)
    
    return dataset_add_auto(original_dataset,pseudo_labeled_dataset)


def sentence_add_manually(dataset):
    columns = ['문장','여성/가족','남성','성소수자','인종/국적','연령','지역','종교','기타 혐오','악플/욕설','clean','개인지칭']
    new_sentence = []
    
    for col in columns:
        if col =='문장':
            new_sentence.append(input(f'{col} :'))
        else:
            new_sentence.append(int(input(f'{col} :'))) 
    
    sentence = pd.DataFrame(columns=columns)
    sentence.loc[0] = new_sentence
    return pd.concat([dataset,sentence],ignore_index=True)

def sentence_add_auto(dataset,new_sentence):
    columns = ['문장','여성/가족','남성','성소수자','인종/국적','연령','지역','종교','기타 혐오','악플/욕설','clean','개인지칭']
    sentence = pd.DataFrame(columns=columns)
    sentence.loc[0] = new_sentence
    return pd.concat([dataset,sentence],ignore_index=True)

def dataset_add_auto(dataset,new_dataset):
    return pd.concat([dataset,new_dataset],ignore_index=True)
    
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

def main():    
    MODEL_NAME = "klue/roberta-large"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #이 부분만 데이터 바꿔 주면 된다
    # pseudo_sentences => unlabeled된 sentences들(Series형태)
    pseudo = pd.read_csv("./Dataset/unsmile_train_v1.0.tsv", sep="\t")
    pseudo_sentences = pseudo["문장"]

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    PATH = "./model_fp/Roberta3.pth"
    model.load_state_dict(torch.load(PATH))
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_psuedo = tokenizer(
                        list(pseudo_sentences), 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=256, 
                        add_special_tokens=True,
                    )

    new_data = pseudo_labeling(pseudo_sentences)

    new_train = dataset_add_auto(pseudo,new_data)

    tokenized_new_train = tokenizer(
                        list(new_train["문장"]), 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=256, 
                        add_special_tokens=True
                    )
    train_new_label = new_train['clean'].values
    train_new_dataset = myDataset(tokenized_new_train, train_new_label)

    training_args = TrainingArguments(
        output_dir="./output9",
        num_train_epochs=10,
        per_device_eval_batch_size=64,
        per_device_train_batch_size=32,
        learning_rate=5e-6,
        logging_dir="./log",
        save_steps=1000,
        save_total_limit=2,
        weight_decay=0.01,
    )

    LENGTH = int(len(train_new_dataset))
    train_data, val_data = random_split(train_new_dataset, [int(LENGTH*0.8), LENGTH-int(LENGTH*0.8)])

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
    )
                
    trainer.train()
    test = pd.read_csv("./Dataset/unsmile_valid_v1.0.tsv", sep="\t")
    tokenized_test = tokenizer(
                        list(test["문장"]), 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=256, 
                        add_special_tokens=True
                    )
    test_label = test['clean'].values
    testdataset = myDataset(tokenized_test, test_label)
    trainer.evaluate(eval_dataset=testdataset)

if __name__=="__main__":
    main()