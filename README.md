#  Cover-Letter Guide 

## Participant 

정영섭: Professor  
권도윤: Student  
박명석: Student  
서민지: Student

## Introduce

취업을 준비하는 과정에서 가장 기본적으로 필수로 준비해야 되는 것은 바로 자기소개서 입니다. 취준생들은 자기소개서를 작성하는데 있어서 많은 어려움을 겪고 있습니다. 이에 본 프로젝트에서는 합격 자기소개서를 바탕으로 분석하여 취준생들에게 자기소개서 작성에 관한 도움을 주고자 합니다. 우리가 제공하는 기능은 다음과 같습니다.  

#### **(1) 특정 분야 및 기업에 관한 키워드**  
자신이 원하는 분야 및 기업에서 합격을 이루었던 중요한 단어들이 무엇인지 파악하여 자기소개서 작성 시 해당 키워드를 참고하여 반영할 수 있도록 합니다.   

#### **(2) 막힌 단어 추천 제공**
자기소개서 작성 시 단어 선택에 있어서 고민이 되는 경우가 많습니다. 막힌 단어와 비슷한 단어들을 추천하여 단어 선택에 관한 고민을 해결해 줄 수 있습니다.

#### **(3) 자기소개서 중요 문장 제공**
자기소개서 안에서 중요한 문장을 제공함으로써 자신이 중요하게 봐야 할 문장을 파악함으로써 면접준비 시 해당 문장을 집중적으로 공부할 수 있다? 아닌가 

#### **(4) 소제목 작성 제공**
실제 인사당담자들은 선호하는 자기소개서 유형으로 "소제목이 있는 자기속서" 를 뽑았습니다.(http://news.bookdb.co.kr/bdb/IssueStory.do?_method=detail&sc.webzNo=37399&Nnews) 키워드 입력시 소제목을 작성해 주는 기능을 통해 소제목 작성에 관한 고민을 해결해주며 현재 취업 트렌드에 맞는 자기소개서를 작성할 수 있도록 도움을 줄 수 있습니다

## Project structure
```
|--Cralwer
|  |--crawler_jobkorea1.py          # jobkorea 사이트 합격 자소서 크롤링 1 
|  |--crawler_jobkorea1.py          # jobkorea 사이트 합격 자소서 크롤링 2  
  
    |--Preprocessing
    |  |--job_answer_preprocess.py     # 데이터 전처리 1
    |  |-- spec_preprocess.py    # 데이터 전처리 2

        |--jobkorea_all.csv     # 전처리 완료 데이터
        |--dataset.txt     # 소제목 데이터

        |--Hannanum
           |--hannanum_().py     # 학과 관련 분야 키워드 워드클라우드(Hannanum 사용)

        |--Komoran
           |--komoran_().py     # 학과 관련 분야 키워드 워드클라우드(Komoran 사용)

        |--Okt
           |--okt_().py      # 학과 관련 분야 키워드 워드클라우드(Okt 사용)

        |--Okt_FastText
           |--okt_FastText_model.py     # FastText 모델(Okt.morphs 사용)
           |--okt_nouns_FastText_model.py     # FastText 적용(Okt.nouns 사용)

        |--Okt_Word2Vec
           |--okt_Word2Vec_model.py    # Word2Ve 모델(Okt.morphs 사용)
           |-- okt_nouns_word2vec_model.py     # Word2Ve 적용(Okt.nouns 사용)

        |--LDA
           |--Hannanum_LDA.py     # LDA 모델(Hannanum 사용)
           |--komoran_LDA.py     # LDA 모델(Komoran 사용)
           |--okt_LDA.py     # LDA 모델(Okt 사용)

        |--TextRank
           |--Hannanum_TextRank.py     # TextRank 모델(Hannanum 사용)
           |--komoran_TextRank.py     # TextRank 모델(Komoran 사용)
           |--okt_TextRank.py     # TextRank 모델(Okt 사용)

        |--SentenceGenerator
           |--RNN_SentenceGenerator.py     # RNN을 사용한 문장 생성 모델
           |--LSTM_SentenceGenerator.py    # LSTM을 사용한 문장 생성 모델
           |--kogpt2_train_SentenceGenerator.py
           |--kogpt2_SenteceGenerator.py

```
## Usage model

![image](/uploads/314785b1187b4d5f88dc17f09813e14a/image.png)

## How to use
   
1) [crawler](https://gitlab.com/DOYUN_K/2020_text_mining/-/tree/master/Crawler, "cralwer link")를 통해 데이터를 수집
2) [Preoprocessing](https://gitlab.com/DOYUN_K/2020_text_mining/-/tree/master/Preprocessing, "preprocessing link")으로 데이터 전처리 진행
3) 원하는 모델(LDA, LSTM 등)선택 후 py파일 실행

### Requirements
```
- tensorflow == 1.14.0
- gensim == 3.8.3
- konlpy == 0.5.2
- wordcloud == 1.8.1
- bs4 == 4.6.0
- Python >= 3.6
- PyTorch == 1.5.0
- MXNet == 1.6.0
- onnxruntime == 1.5.2
- gluonnlp == 0.9.1
- sentencepiece >= 0.1.85
- transformers == 2.11.0
```

* LDA 모델을 사용하기 위해서는 MALLET 패키지 다운이 필요합니다. 해당 모델을 사용할 경우 [MALLET](http://mallet.cs.umass.edu/index.php, "MALLET") 에서   
  다운 받아 사용하여 주세요.
   
* 저희는 SKT-AI에서 약 20GB의 한국어 데이터를 Pre-Training 시킨 [KoGPT2](https://github.com/SKT-AI/KoGPT2,"kogpt2") 모델을 사용하였습니다. 따라서 
   KoGPT2 사용을 위해서는 아래와 같은 작업 및 설치가 필요합니다.
  

   ### Kogpt2 How to install
   ```
      git clone https://github.com/SKT-AI/KoGPT2.git
      pip install -r requirements.txt
      pip install .
   ```
      
## License

KOGPT2와 LDA는 modified MIT 라이선스 하에 공개되어 있습니다. 해당 모델및 코드를 사용할 경우 모델 및 코드를 사용할 경우 라이선스 내용을 준수해주세요. 라이선스 내용은 [LICENSE](https://gitlab.com/jaaaamj0711/example/-/blob/master/LICENSE, "LICENSE") 파일에서 확인할 수 있습니다.  
