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
