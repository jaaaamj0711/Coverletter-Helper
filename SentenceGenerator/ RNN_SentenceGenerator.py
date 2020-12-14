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

