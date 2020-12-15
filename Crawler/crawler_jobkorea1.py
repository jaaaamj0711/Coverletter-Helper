import pandas as pd
import re
import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup

# 회사명, 지원시기, 근무형태, 직무분야, 주소 가져오기 
def jobkorea1(pages: int):
    
    assert pages > 0
    
    url_basic = "http://www.jobkorea.co.kr/starter/PassAssay?FavorCo_Stat=0&Pass_An_Stat=0&OrderBy=0&EduType=0&WorkType=0&isSaved=0&Page="
    result = pd.DataFrame()
    
    for page in range(pages):
        url = url_basic + str(page + 1)
        
        try:
            html = urlopen(url)
            soup = BeautifulSoup(html, "lxml")
            
            
            company = soup.findAll("span", attrs={"class": "titTx"})
            company = list(map(lambda x: x.text, company))
            
            career = soup.findAll("span", attrs={"class": "career"})
            career = list(map(lambda x: x.text, career))

            field = soup.findAll("span", attrs={"class": "field"})
            
            field_1 = list(map(lambda x: x.text, field[::2]))
           
            field_2 = list(map(lambda x: x.text, field[1::2]))

            urls = soup.findAll("ul", attrs={"class": "selfLists"})[0].findAll("li")
            urls = list(map(lambda x: "http://www.jobkorea.co.kr" + x.find("a")["href"], urls))
            
            temp = pd.DataFrame({"회사명": company, "지원시기": career, "근무형태": field_1, 
                                 "직무분야": field_2, "주소": urls})
            result = result.append(temp)
            
        except:
            continue
                
    return result.reset_index(drop=True)