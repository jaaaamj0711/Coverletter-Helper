import pandas as pd
from string import punctuation
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('./2020_text_mining/kogpt2/dataset.txt')
df.head()
df['제목'].isnull().values.any()