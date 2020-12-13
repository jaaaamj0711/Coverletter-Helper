import pandas as pd
from string import punctuation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('./2020_text_mining/kogpt2/dataset.txt')
df.head()
df['제목'].isnull().values.any()

headline = [] # 리스트 선언
headline.extend(list(df.제목.values)) # 헤드라인의 값들을 리스트로 저장
headline[:5] # 상위 5개만 출력

print('총 샘플의 개수 : {}'.format(len(headline))) # 현재 샘플의 개수

headline = [n for n in headline if n != "Unknown"] # Unknown 값을 가진 샘플 제거
print('노이즈값 제거 후 샘플의 개수 : {}'.format(len(headline))) # 제거 후 샘플의 개수
