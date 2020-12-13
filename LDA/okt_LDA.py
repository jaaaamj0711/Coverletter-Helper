import numpy as np
import pandas as pd

data = pd.read_csv("jobkorea_data.csv")

data.shape
data.columns.tolist()

from konlpy.tag import Okt  
okt=Okt()

x_list = data['답변']
x_list

data_word=[]
for i in range(len(x_list)):
    try:
        data_word.append(okt.nouns(x_list[i]))
    except Exception as e:
        continue

Data_list=x_list.values.tolist()

from gensim import corpora, models
from gensim.models.wrappers import LdaMallet

id2word=corpora.Dictionary(data_word)
id2word.filter_extremes(no_below = 0) #20회 이하로 등장한 단어는 삭제

texts = data_word
corpus=[id2word.doc2bow(text) for text in texts]

mallet_path = '/Users/doyun/Downloads/mallet-2.0.8/bin/mallet' 
ldamallet = models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=10, id2word=id2word)

from gensim.models.coherencemodel import CoherenceModel

coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()

