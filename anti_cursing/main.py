import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils import *

def main():
    while True:
        # anti = anti_cursing()
        sent = input("문장을 입력하세요: ")
        if sent == "0":
            break
        
        logging.info("please wait for a second")
        # print(main(sent, anti))
        an = antiCursing()
        print(an.anti_cur(sentence = sent))
        print("-"*50)   
        
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading model")
    logging.info("run main function")
    
    main()
    