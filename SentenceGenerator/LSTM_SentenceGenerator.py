import pandas as pd
import numpy as np
from string import punctuation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM

# 데이터 불러오기
df = pd.read_csv('./2020_text_mining/kogpt2/dataset.txt')

# 데이터 확인
df.head()
df['제목'].isnull().values.any()

# Unknown 값을 가진 샘플 제거 및 샘플 수 확인
headline = [n for n in headline if n != "Unknown"]
print('노이즈값 제거 후 샘플의 개수 : {}'.format(len(headline)))

text = headline

t = Tokenizer()
t.fit_on_texts(text)

# 단어 집합 크기 확인
vocab_size = len(t.word_index) + 1
print('단어 집합의 크기 : %d' % vocab_size)

# 인코딩
sequences = list()
for line in text: 
    encoded = t.texts_to_sequences([line])[0] 
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

# 시퀀스 예시 출력
sequences[:11]

# 인덱스를 단어로 바꾸기 위해 index_to_word를 생성
index_to_word = {}
for key, value in t.word_index.items(): 
    index_to_word[value] = key

# 빈도수 상위 Top 10 확인
print('빈도수 상위 10번 단어 : {}'.format(index_to_word[10]))

# 샘플 최대 길이 확인
max_len = max(len(l) for l in sequences)
print('샘플의 최대 길이 : {}'.format(max_len))

# Padding
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
print(sequences[:3])

sequences = np.array(sequences)

# Data split
X = sequences[:,:-1]
y = sequences[:,-1]

y = to_categorical(y, num_classes=vocab_size)

# Modeling
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=24, verbose=2)

# Sentence Generator 정의
def sentence_generation(model, t, current_word, n): 
    init_word = current_word 
    sentence = ''
    for _ in range(n):
        encoded = t.texts_to_sequences([current_word])[0] 
        encoded = pad_sequences([encoded], maxlen=627, padding='pre') 
        result = model.predict_classes(encoded, verbose=0)
    
        for word, index in t.word_index.items(): 
            if index == result: 
                break 
        current_word = current_word + ' '  + word 
        sentence = sentence + ' ' + word 
  
    sentence = init_word + sentence
    return sentence

# 결과확인
print(sentence_generation(model, t, '개발', 3))
print(sentence_generation(model, t, '분석', 3))
print(sentence_generation(model, t, '데이터', 3))