import pandas as pd
import re
import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup

# url 받아오는 함수
def saramin1(pages: int):
    
    assert pages > 0
    
    url_base = "http://www.saramin.co.kr/zf_user/public-recruit/coverletter-list/page/"
    
    result = pd.DataFrame()
    
    for page in range(pages):
        
        url = url_base + str(page + 1)
        
        try:
            html = urlopen(url)
            soup = BeautifulSoup(html, "lxml")
            
            table = pd.read_html(url)[0]
            
            urls = soup.findAll("td", attrs={"class": "td_apply_subject"})
           
            urls = list(map(lambda x: "http://www.saramin.co.kr" + x.find("a")["href"], urls))
            
            table["주소"] = urls
            
            result = result.append(table)
            
        except:
            continue
        
    return result.reset_index(drop=True)

# 질문과 답변 받아오는 함수
def saramin2(column="주소"):
    
    saramin1 = pd.read_csv("./saramin1.csv")
    
    result = pd.DataFrame()
    
    for i in range(saramin1.shape[0]):
        
        url = saramin1.loc[i, column]
        
        try:
            html = urlopen(url)
            soup = BeautifulSoup(html, "lxml")
            
            questions = soup.findAll("div", attrs={"class": "item_self"})
            questions = list(map(lambda x: x.find("h3").text, questions))
            
            answers = soup.findAll("div", attrs={"class": "box_ty3"})
            answers = list(map(lambda x: x.text, answers))
            
            temp = pd.DataFrame({"질문": questions, "답변": answers})
            temp["주소"] = url
            
            result = result.append(temp)
            
        except:
            continue
    
    return result.reset_index(drop=True)