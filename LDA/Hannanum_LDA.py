import numpy as np
import pandas as pd

data = pd.read_csv("jobkorea_data.csv")

data.shape
data.columns.tolist()

from konlpy.tag import Hannanum  
hannanum=Hannanum()

x_list = data['답변']
x_list

data_word=[]
for i in range(len(x_list)):
    try:
        data_word.append(hannanum.nouns(x_list[i]))
    except Exception as e:
        continue

Data_list=x_list.values.tolist()