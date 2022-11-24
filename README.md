# anti-cursing

**"anti-cursing"** is a python package that detects and switches negative or any kind of cursing word from sentences or comments whatever🤬

You just install the package the way you install any other package and then you can use it in your code.

The whole thing is gonna be updated soon.

So this is __**the very first idea**__

But you can find my package in pypi(https://pypi.org/project/anti-cursing/0.0.1/)

**🙏🏻Plz bare with the program to install model's weight and bias from huggingface at the first time you use the package.**

<img width="1134" alt="image" src="https://user-images.githubusercontent.com/50198431/203723736-3aeb84a1-6418-4190-b967-2888e14b14fd.png">

<hr>

# Concept
There are often situations where you have to code something, detect a forbidden word, and change it to another word.
Hardcoding all parts is very inconvenient, and in the Python ecosystem, there are many packages to address.
One of them is **"anti-cursing"**.

The package, which operates exclusively for **Korean**, does not simply change the banned word by setting it up, but detects and replaces the banned word by learning a deep learning model.

Therefore, it is easy to cope with new malicious words as long as they are learned.
For this purpose, semi-supervied learning through pseudo labeling is used.

Additionally, instead of changing malicious words to special characters such as --- or ***, you can convert them into emojis to make them more natural.

# Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model comparison](#model-comparison)
- [Dataset](#dataset)
- [Used API](#used-api)
- [License](#license)
- [Working Example](#working-example)
- [References](#references)
- [Project Status](#project-status)
- [Future Work](#future-work)


# Installation

You can install the package using pip:

```bash
pip install anti-cursing
```

**it doesn't work yet, but it will soon!!👨🏻‍💻**

# Usage

```python
from anti_cursing.utils import antiCursing

antiCursing.anti_cur("나는 너가 좋지만, 너는 너무 개새끼야")
```

```bash
나는 너가 좋지만, 너는 너무 👼🏻야
```

# Model-comparison
| Classification | KcElectra | KoBERT | RoBERTa-base | RoBERTa-large |
| --- | --- | --- | --- | --- |
| Validation Accuracy | 0.88680 | 0.85721 | 0.83421 | 0.86994 |
| Validation Loss | 1.00431 | 1.23237 | 1.30012 | 1.16179 |
| Training Loss | 0.09908 | 0.03761 | 0.0039 | 0.06255 |
| Epoch | 10 | 40 | 20 | 20 |
| Batch-size | 8 | 32 | 16 | 32 |
| transformers | beomi/KcELECTRA-base | skt/kobert-base-v1 | xlm-roberta-base | klue/roberta-large |

# Dataset
* ### Smilegate-AI 
  * https://github.com/smilegate-ai/korean_unsmile_dataset
  * Korean Sentiment Analysis
  * [paper](#korean-unsmile-dataset)
* ### Naver portal news articles crawling
  * https://news.naver.com
  * Non-labeled Data for Test Dataset
* ### 😀 Emoji unicode crawling for encoding
  * https://unicode.org/emoji/charts/full-emoji-list.html

# Used-api
### Google translator
* https://cloud.google.com/translate/docs (API DOCS)

# License

This repository is licensed under the MIT license. See LICENSE for details.

Click here to see the License information --> [License](LICENSE)

# Working-example
---- some video is gonna be placed here ----

# References

### Sentiment Analysis Based on Deep Learning : A Comparative Study
  * Nhan Cach Dang, Maria N. Moreno-Garcia, Fernando De la Prieta. 2006. Sentiment Analysis Based on Deep Learning : A Comparative Study. In Proceedings of the 2006 Conference on Empirical Methods in Natural Language Processing (EMNLP 2006), pages 1–8, Prague, Czech Republic. Association for Computational Linguistics.
### Attention is all you need
  * Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, pages 6000–6010.
### BERT : Pre-training of Deep Bidirectional Transformers for Language Understanding
  * Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT:         Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 4171–4186.

### Electra : Pre-training Text Encoders as Discriminators Rather Than Generators
  * Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning. 2019. Electra: Pre-training text encoders as discriminators rather than generators. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 4171–4186.

### BIDAF : Bidirectional Attention Flow for Machine Comprehension
  * Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi. 2016. Bidirectional Attention Flow for Machine Comprehension. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2129–2139.

### Effect of Negation in Sentences on Sentiment Analysis and Polarity Detection
  * Partha Mukherjeea, Saptarshi Ghoshb, and Saptarshi Ghoshc. 2018. Effect of Negation in Sentences on Sentiment Analysis and Polarity Detection. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2129–2139.

### KOAS : Korean Text Offensiveness Analysis System
  * Seonghwan Kim, Seongwon Lee, and Seungwon Do. 2019. KOAS: Korean Text Offensiveness Analysis System. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1–11.

### Korean Unsmile Dataset
  * Seonghwan Kim, Seongwon Lee, and Seungwon Do. 2019. Korean Unsmile Dataset. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1–11.
# Project-status

![80%](https://geps.dev/progress/80)

# Future-work
update soon plz bare with me 🙏🏻

<hr>

# KOREAN FROM HERE / 여기부턴 한국어 설명입니다.
# anti-cursing

**"anti-cursing"**은 문장이나 댓글에서 부정적이거나 모든 종류의 욕설을 감지하고 전환하는 파이썬 패키지입니다🤬

다른 패키지를 설치하는 방식과 동일하게 패키지를 설치한 다음 코드에서 사용할 수 있습니다.

아직 아이디어 구상 단계이기 때문에 **아무것도 작동하지 않지만** 곧 작동하도록 업데이트할 예정입니다.

Pypi(https://pypi.org/project/anti-cursing/0.0.1/)에 패키지르 업로드했습니다. 확인하시 수 있습니다.

**🙏🏻패키지를 처음 설치하시고 사용하실 때 딥러닝 모델을 불러오기 위해 huggingface에서 parsing을 시도합니다. 처음에만 해당 작업이 필요하니 시간이 조금 걸림과 용량을 차지함을 고려해주세요**

<img width="1134" alt="image" src="https://user-images.githubusercontent.com/50198431/203723736-3aeb84a1-6418-4190-b967-2888e14b14fd.png">


<hr>

# Concept

무언가 코딩을 하며, 금지 단어를 감지하고 그것을 다른 단어로 바꿔야할 상황이 종종 생깁니다.
모든 부분을 하드코딩하는 것이 매우 불편하며, 파이썬 생태계에서는 이를 해결하기 위한 많은 패키지가 있습니다.
그 중 하나가 **"anti-cursing"**입니다.

한국어 전용으로 동작하는 해당 패키지는 단순히 금지 단어를 기존에 설정하여 바꾸는 것이 아닌, 딥러닝 모델을 학습하여 금지 단어를 감지하고 바꿉니다.
따라서 새롭게 생기는 악성 단어에 대해서도 학습만 이루어진다면 쉽게 대처할 수 있습니다.
이를 위해 pseudo labeling을 통한 semi-supervied learning을 사용합니다. 

추가로 악성단어를 ---나 ***같은 특수문자로 변경하는 것이 아닌, 이모지로 변환하여 더욱 자연스럽게 바꿀 수 있습니다.
# 목차

- [설치](#설치)
- [사용법](#사용법)
- [모델 성능 비교](#모델-성능-비교)
- [데이터셋](#데이터셋)
- [사용 API](#사용-api)
- [License](#license)
- [작동 예시](#작동-예시)
- [참고문헌](#참고문헌)
- [진행상황](#진행상황)
- [발전](#발전)


# 설치

pip를 사용하여 패키지를 설치할 수 있습니다.
```bash
pip install anti-cursing
```

**아직 아무것도 작동하지 않지만, 곧 작동하도록 업데이트할 예정입니다👨🏻‍💻.**

# 사용법

```python
from anti_cursing.utils import antiCursing

antiCursing.anti_cur("나는 너가 좋지만, 너는 너무 개새끼야")
```

```bash
나는 너가 좋지만, 너는 너무 👼🏻야
```

# 모델 성능 비교
| Classification | KcElectra | KoBERT | RoBERTa-base | RoBERTa-large |
| --- | --- | --- | --- | --- |
| Validation Accuracy | 0.88680 | 0.85721 | 0.83421 | 0.86994 |
| Validation Loss | 1.00431 | 1.23237 | 1.30012 | 1.16179 |
| Training Loss | 0.09908 | 0.03761 | 0.0039 | 0.06255 |
| Epoch | 10 | 40 | 20 | 20 |
| Batch-size | 8 | 32 | 16 | 32 |
| transformers | beomi/KcELECTRA-base | skt/kobert-base-v1 | xlm-roberta-base | klue/roberta-large |


# 데이터셋
* ### Smilegate-AI 
  * https://github.com/smilegate-ai/korean_unsmile_dataset
  * 한국어 감정분류 데이터셋
  * [paper](#korean-unsmile-dataset)
* ### 네이버 뉴스 기사 크롤링
  * https://news.naver.com
  * 테스트를 위한 데이터셋
* ### 😀 이모지 유니코드 데이터셋
  * https://unicode.org/emoji/charts/full-emoji-list.html

# 사용 API
### Google translator
* https://cloud.google.com/translate/docs (API 문서)

# License

이 프로젝트는 MIT 라이센스를 따릅니다. 자세한 내용은 LICENSE 파일을 참고해주세요.

라이센스 정보 --> [License](LICENSE)

# 작동 예시
---- 작동 예시가 추가될 예정입니다 ----

# 참고문헌

### Sentiment Analysis Based on Deep Learning : A Comparative Study
  * Nhan Cach Dang, Maria N. Moreno-Garcia, Fernando De la Prieta. 2006. Sentiment Analysis Based on Deep Learning : A Comparative Study. In Proceedings of the 2006 Conference on Empirical Methods in Natural Language Processing (EMNLP 2006), pages 1–8, Prague, Czech Republic. Association for Computational Linguistics.
### Attention is all you need
  * Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, pages 6000–6010.
### BERT : Pre-training of Deep Bidirectional Transformers for Language Understanding
  * Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT:         Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 4171–4186.

### Electra : Pre-training Text Encoders as Discriminators Rather Than Generators
  * Kevin Clark, Minh-Thang Luong, Quoc V. Le, Christopher D. Manning. 2019. Electra: Pre-training text encoders as discriminators rather than generators. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 4171–4186.

### BIDAF : Bidirectional Attention Flow for Machine Comprehension
  * Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi. 2016. Bidirectional Attention Flow for Machine Comprehension. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2129–2139.

### Effect of Negation in Sentences on Sentiment Analysis and Polarity Detection
  * Partha Mukherjeea, Saptarshi Ghoshb, and Saptarshi Ghoshc. 2018. Effect of Negation in Sentences on Sentiment Analysis and Polarity Detection. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2129–2139.

### KOAS : Korean Text Offensiveness Analysis System
  * Seonghwan Kim, Seongwon Lee, and Seungwon Do. 2019. KOAS: Korean Text Offensiveness Analysis System. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1–11.

### Korean Unsmile Dataset
  * Seonghwan Kim, Seongwon Lee, and Seungwon Do. 2019. Korean Unsmile Dataset. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 1–11.
# 진행상황

![80%](https://geps.dev/progress/80)

# 발전
앞으로 추가될 예정입니다 잠시만 기다려주세요🙏🏻
