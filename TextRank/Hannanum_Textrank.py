from konlpy.tag import Kkma
from konlpy.tag import Hannanum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from preprocess_all import preprocess_answer
import numpy as np
import pandas as pd


data = pd.read_csv("./2020_text_mining/jobkorea_data.csv")

data = preprocess_answer(data)

# 문장 토큰화
class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        self.hannanum = Hannanum()
    
    # 텍스트를 입력으로 받아서 문장단위로 나누어 줌
    def text_sentences(self, text):
        sentences = self.kkma.sentences(text)
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx-1] += (' ' + sentences[idx])
                sentences[idx] = ''
        return sentences

        # 문장 단위로 입력을 받아서 명사를 출력
    def sentences_nouns(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentence is not '':
                nouns.append(' '.join([noun for noun in self.hannanum.nouns(str(sentence))
                                if len(noun) > 1]))
        return nouns 

# TF-IDF 생성
class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []
    def sentence_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T) # TF-IDF matrix 
        return self.graph_sentence # Sentence graph

