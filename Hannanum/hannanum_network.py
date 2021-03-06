from wordcloud import WordCloud
from collections import Counter
from re import match
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from konlpy.tag import Hannanum  

data = pd.read_csv("/Users/doyun/2020_text_mining/jobkorea_data.csv")

hannanum = Hannanum()

ap_progra = data.loc[data['직무분야'] == "네트워크·서버·보안", "답변"]
nouns = hannanum.nouns(''.join(str(ap_progra.fillna(''))))
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
plt.imshow(구름)
