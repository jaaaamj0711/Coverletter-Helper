from konlpy.tag import Kkma
from konlpy.tag import Hannanum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from preprocess_all import preprocess_answer
import numpy as np
import pandas as pd


data = pd.read_csv("./jobkorea_all.csv")

data = preprocess_answer(data)
