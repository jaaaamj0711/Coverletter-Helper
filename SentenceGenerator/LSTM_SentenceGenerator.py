import pandas as pd
import numpy as np
from string import punctuation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# 데이터 불러오기
df = pd.read_csv('./2020_text_mining/kogpt2/dataset.txt')

# 데이터 확인
df.head()
df['제목'].isnull().values.any()

headline = [n for n in headline if n != "Unknown"] # Unknown 값을 가진 샘플 제거
print('노이즈값 제거 후 샘플의 개수 : {}'.format(len(headline))) # 제거 후 샘플의 개수

text = headline

t = Tokenizer()
t.fit_on_texts(text)

vocab_size = len(t.word_index) + 1
print('단어 집합의 크기 : %d' % vocab_size)

sequences = list()
for line in text: # 1,214 개의 샘플에 대해서 샘플을 1개씩 가져온다.
    encoded = t.texts_to_sequences([line])[0] # 각 샘플에 대한 정수 인코딩
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

# 시퀀스 예시 출력
sequences[:11]

