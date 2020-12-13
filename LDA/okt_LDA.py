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
