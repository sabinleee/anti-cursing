from utils import *

def main(sent, anti):
    triger = 0
    new_comment = ""
    for sent in anti.split_sentence(sent):
        batch_sentence_prediction, updated_trgier = anti.sentence_predict(sent, triger)
        
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading model")
    logging.info("run main function")
    
    while True:
        anti = anti_cursing()
        sent = input("문장을 입력하세요: ")
        if sent == "0":
            break
        
        logging.info("please waith for a second")
        print(main(sent, anti))
        print("-"*50)   
