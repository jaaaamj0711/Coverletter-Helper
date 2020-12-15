from wordcloud import WordCloud
from collections import Counter
from re import match
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from konlpy.tag import Okt  

data = pd.read_csv("/Users/doyun/2020_text_mining/jobkorea_data.csv")

okt = Okt()

ap_progra = data.loc[data['직무분야'] == "응용프로그래머", "답변"]
nouns = okt.nouns(''.join(str(ap_progra.fillna(''))))
nouns = [n for n in nouns if len(n) > 1]
nouns = [n for n in nouns if not(match('^[0-9]',n))]
count = Counter(nouns)
top = count.most_common(40)

my_font_path = '/Users/doyun/Library/Fonts/NanumBarunGothic.ttf'

wordcloud = WordCloud(font_path=my_font_path, 
                    background_color='white', width=800, height=600)
cloud = wordcloud.generate_from_frequencies(dict(top))
plt.figure(figsize=(10, 8))
plt.axis('off')
plt.imshow(cloud)