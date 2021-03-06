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
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T) 
        return self.graph_sentence 

# TextRank 알고리즘을 구현
class Rank(object):
    def get_ranks(self, graph, d=0.85): 
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0 
            link_sum = np.sum(A[:,id])
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1
        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B) 
        return {idx: r[0] for idx, r in enumerate(ranks)}

# TextRank 문장 요약
class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()
        self.sentences = self.sent_tokenize.text_sentences(text)
        self.nouns = self.sent_tokenize.sentences_nouns(self.sentences)
        self.graph_matrix = GraphMatrix()
        self.sent_graph = self.graph_matrix.sentence_graph(self.nouns)
        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)

     # 3줄 요약
    def summarize(self, sent_num=3): 
        summary = []
        index=[]
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        index.sort()
        for idx in index:
            summary.append(self.sentences[idx])
        return summary

# 결과 보기
t = data['답변'][100]
textrank = TextRank(t)

for row in textrank.summarize(3):
    print(row)