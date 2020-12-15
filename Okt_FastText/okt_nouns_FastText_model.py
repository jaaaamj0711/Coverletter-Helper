import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
okt = Okt()

data = pd.read_csv("jobkorea_data.csv")

data['답변'] = data['답변'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# 토큰화 및 불용어 제거
tokenized_data = []
for sentence in data['답변']:
    temp_X = okt.nouns(sentence) 
    temp_X = [word for word in temp_X if not word in stopwords] 
    tokenized_data.append(temp_X)


# 분포 확인
print('최대 길이 :', max(len(l) for l in tokenized_data))
print('평균 길이 :', sum(map(len, tokenized_data))/len(tokenized_data))
plt.hist([len(s) for s in tokenized_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

model_ft = FastText(tokenized_data, size=100, workers=4, sg=1, iter=6, word_ngrams=5)

model_ft_df = pd.DataFrame(model_f.wv.most_similar("데이터"), columns=['단어', '유사도'])

print("선택 단어 : {}".format("데이터"))
model_ft_df