import pandas as pd
import re
import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup

# 질문, 답변, 조언, 스펙, 평가, 총평 가져오기
def jobkorea2(column="주소"):
    
    jobkorea1 = pd.read_csv("./jobkorea1.csv")
    result = pd.DataFrame()
    
    for i in range(jobkorea1.shape[0]):
        url = jobkorea1.loc[i, column]
        
        try:
            html = urlopen(url)
            soup = BeautifulSoup(html, "lxml")
            
            speclists = soup.findAll("ul", attrs={"class": "specLists"})[0].text
            
            grade = soup.findAll("span", attrs={"class": "grade"})[0].text
            
            advice_Total = soup.findAll("div",
                                        attrs={"class": "adviceTotal"})[0].findAll("p", attrs={"class": "tx"})[0].text
            
            questions = list(map(lambda x: x.text,
                                 soup.findAll("dl", attrs={"class": "qnaLists"})[0].findAll("span", attrs={"class": "tx"})))
            
            answers = list(map(lambda x: x.text,
                               soup.findAll("dl", attrs={"class": "qnaLists"})[0].findAll("div", attrs={"class": "tx"})))
            
            advices = soup.findAll("dd", attrs={"class": "show"})
            
            advices = list(map(lambda x: x.findAll("div",
                                                   attrs={"class": "advice"})[0].text, advices))
            
            for j in range(len(answers) - len(advices)):
                advice = soup.findAll("dd",
                                      attrs={"class": ""})[-(j+1)].findAll("div", attrs={"class": "advice"})[0].text
                
                advices.append(advice)
                
            sup = pd.DataFrame({"질문": questions, "답변": answers, "조언": advices})
            sup["스펙"] = speclists
            sup["평가"] = grade
            sup["총평"] = advice_Total
            sup["주소"] = url
            
            result = result.append(sup)
                
        except:
            continue
    
    return result.reset_index(drop=True)