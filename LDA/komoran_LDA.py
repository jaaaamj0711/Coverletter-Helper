import numpy as np
import pandas as pd

data = pd.read_csv("C:/Users/user/Desktop/2020_text_mining/jobkorea_data.csv")

from konlpy.tag import Komoran
komoran=Komoran()

x_list = data['답변']
x_list