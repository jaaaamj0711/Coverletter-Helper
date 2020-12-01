# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 16:54:35 2020

@author: user
"""

import numpy as np
import pandas as pd
from collections import Counter
from re import match


data = pd.read_csv("C:/Users/user/Desktop/2020_text_mining/jobkorea_data.csv")


data['직무분야'].unique()
komoran = Komoran()

%time komoran_nouns = komoran.nouns(''.join(str(data['답변'].fillna(''))))
komoran_nouns[-10:]

DBA = data.loc[data['직무분야'] == "시스템프로그래머", "답변"]
nouns = komoran.nouns(''.join(str(DBA.fillna(''))))
nouns = [n for n in nouns if len(n) > 1]
nouns = [n for n in nouns if not(match('^[0-9]',n))]
count = Counter(nouns)
top = count.most_common(40)# 그 여기가 이슈 3에서 다루는 부분

#불용어 제거 
stopwords=set(STOPWORDS)

stopwords.add('제가')

wordcloud = WordCloud(font_path='C:/Users/user/Desktop/2020_text_mining/NanumGothic.ttf', 
                   background_color='white', width=800, height=600,stopwords=stopwords)


cloud = wordcloud.generate_from_frequencies(dict(top))


plt.figure(figsize=(10,8))
plt.imshow(wordcloud)#TODO  
plt.tight_layout(pad=0)
plt.axis('off')
plt.show()