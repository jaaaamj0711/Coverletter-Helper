import numpy as np
import pandas as pd

data = pd.read_csv("C:/Users/user/Desktop/2020_text_mining/jobkorea_data.csv")

from konlpy.tag import Komoran
komoran=Komoran()

x_list = data['답변']
x_list

data_word=[]
for i in range(len(x_list)):
    try:
        data_word.append(komoran.nouns(x_list[i]))
    except Exception as e:


        continue

Data_list=x_list.values.tolist()


from gensim import corpora, models
from gensim.models.wrappers import LdaMallet

id2word=corpora.Dictionary(data_word)
id2word.filter_extremes(no_below = 0) #20회 이하로 등장한 단어는 삭제

texts = data_word
corpus=[id2word.doc2bow(text) for text in texts]

mallet_path = '/Users/user/Downloads/mallet-2.0.8/bin/mallet' 
ldamallet = models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=10, id2word=id2word)

from gensim.models.coherencemodel import CoherenceModel

coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()


def compute_coherence_values(dictionary, corpus, texts, limit, start=4, step=2):

    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=data_word, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values





