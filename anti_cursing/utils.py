import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from lib import *

class myDataset(torch.utils.data.Dataset):
    def __init__(self, encoding):
        self.encoding = encoding
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encoding.items()}
        return item

    def __len__(self):
        return len(self.encoding['input_ids'])


class antiCursing:
    def __init__(self, BASE_PATH=os.path.abspath(os.path.dirname(__file__))) -> None:
        self.BASE_PATH = BASE_PATH
        self.model = AutoModelForSequenceClassification.from_pretrained("24bean/kcelectra_senti_binary")
        self.multi_model = AutoModelForSequenceClassification.from_pretrained("24bean/multi_classification")
        self.tokenizer = AutoTokenizer.from_pretrained("24bean/tokenizer_KcElectra")
        self.emoji = pd.read_csv(self.BASE_PATH + "/dataset/emoji_category.csv")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.TAG_LIST = ["JKS","JKC","JKG","JKO","JKB","JKV","JKQ","JX","JC","EP","EF","EC","ETN","ETM","SF","SE","SSO","SSC","SC","SY", "VV"]
        self.mmscaler = MinMaxScaler()
        self.columns = ['악플/욕설', '인종/국적', '개인지칭', '연령', '남성', '여성/가족', '성소수자', '지역', '기타 혐오', '종교']

        self.model.eval()
            
    def pass_model(self, token):
        with torch.no_grad():
            output = self.model(**token)

        logits = output[0].detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)

        return result

    def pass_model_for_batch(self, masked_loader):
        # T = 0.95
        self.model.eval()

        masked_outputs=[]
        with torch.no_grad():
            for data in masked_loader:
                outputs = self.model(**data)
                masked_outputs.extend(torch.sigmoid(outputs["logits"]).cpu().detach().numpy().tolist())
            
        return masked_outputs

    def tokenize_list(self, list_to_tokenize=list):
        tokenized_list = self.tokenizer(
                        list_to_tokenize,
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=512, 
                        add_special_tokens=True
                    )
        return tokenized_list


    def word_predict(self, tokenized_sent):
        masked_token = {}
        masked_sentence = []
        where_mask = []
        mask_ids_combination = []
        token_to_mask = []

        
        words = [word for word in tokenized_sent["input_ids"][0]][1:-1]
        token_count = dict(zip(words, [0]*len(words)))
        
        [mask_ids_combination.extend(combinations(words,i)) for i in range(1, len(words)+1)]
        for mask_ids in mask_ids_combination:
            where_mask.append(mask_ids)
            masked_token[mask_ids] = []
            
            for word in tokenized_sent["input_ids"][0]:
                if word in mask_ids:
                    masked_token[mask_ids].append(torch.tensor(4).to(self.device))
                else:
                    masked_token[mask_ids].append(word)

            masked_sentence.append(self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(masked_token[mask_ids])))

        masked_dataset = self.tokenize_list(masked_sentence)
        mask_dataset = myDataset(masked_dataset)
        masked_params = {'batch_size': 8,
                        'shuffle': False,
                        'num_workers': 0
                        }
        masked_loader = DataLoader(mask_dataset, **masked_params)
        masked_outputs = self.pass_model_for_batch(masked_loader)
        
        T = 0.85    # threshold for word prediction
        for temp_output, temp_mask in zip(masked_outputs, where_mask):
            for word in temp_mask:
                if(temp_output[1] > T):
                    token_count[word] += 1 
        scaled_count_values =  self.mmscaler.fit_transform(np.array(list(token_count.values())).reshape(-1,1))

        mmerror = True
        for scaled_count_value in scaled_count_values:
            if scaled_count_value > 0:
                mmerror = False
                break
        if mmerror == True:
            return self.curse_word_to_emoji(self.tokenizer.convert_ids_to_tokens(token_count.keys()))

                
        MMT = 0.5
        [token_to_mask.append(list(token_count.keys())[idx]) for idx in range(len(scaled_count_values)) if scaled_count_values[idx] >= MMT]
        # min-max scale 해서 0.5 이상을 마스킹
        sub_emoji = list(self.curse_word_to_emoji(self.tokenizer.convert_ids_to_tokens(token_to_mask)))
        final_tokens = self.tokenizer.convert_ids_to_tokens(token_count.keys())
        for final_token_idx in range(len(final_tokens)):
            for final_mask in self.tokenizer.convert_ids_to_tokens(token_to_mask):
                if final_tokens[final_token_idx] == final_mask and len(sub_emoji):
                    final_tokens[final_token_idx] = sub_emoji.pop(0)
                
        return final_tokens


    def sentence_multi_classification(self, mask_word):
        mask_word = self.tokenize_list(mask_word)
        self.multi_model.eval()
        
        with torch.no_grad():
            output = self.multi_model(**mask_word)
            
        outputs = output['logits'].detach().cpu().numpy()
        final_outputs = np.array(outputs) >=0.5
        
        return (pd.DataFrame(final_outputs,columns=self.columns))


    def curse_word_to_emoji(self, word):
        random_emoji = []
        final_outputs = self.sentence_multi_classification(word)

        for col in self.columns:
            for output_idx in range(len(final_outputs[col])):
                if final_outputs.loc[output_idx,col] == True:
                        random_emoji.extend(self.emoji.loc[self.emoji["category"] == col, "Browser"].tolist())
        
        if random_emoji == []:
            random_emoji.extend(self.emoji.loc[self.emoji["category"] == '기타', "Browser"].tolist())

        return random.sample(random_emoji,len(final_outputs[col]))[:len(final_outputs[col])]

    def sentence_predict(self, sent, triger):
        tokenized_sent = self.tokenize_list(sent)
        tokenized_sent.to(self.device)
        result = self.pass_model(tokenized_sent)

        # 악성댓글
        if result == 0:
            triger += 1
            return self.word_predict(tokenized_sent), triger
        else:
            return sent, triger

    def split_sentence(self, sent):
        temp = sent.strip().split()
        sent_split = []
        for i in range(0,len(temp),4):
            sent_split.append(' '.join(temp[i:i+4]))

        return sent_split


    def anti_cur(self, sentence):
        triger = 0
        new_comment = ""
        for sent in self.split_sentence(sentence):
            batch_sentence_prediction, updated_trgier = self.sentence_predict(sent, triger)
            
            if type(batch_sentence_prediction) == str:
                new_comment += (" " + batch_sentence_prediction)
                continue

            for word in batch_sentence_prediction:
                if word[:2] == "##":
                    new_comment += word[2:]
                else:
                    new_comment += (" " + word)
        
            triger = updated_trgier

        if triger == 0:
            return(new_comment)
        if triger > 0:
            return(new_comment)
            