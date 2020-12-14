import re
import numpy as np
import pandas as pd

#  str.replace 함수 정의(특정 패턴 공백으로 처리)
def str_replace(text, patterns: list, replace=" "):
    
    assert isinstance(text, pd.core.series.Series)
    assert isinstance(patterns, list)
    
    for pattern in patterns:
        text = text.str.replace(pattern, replace)
    
    return text

def split_spec(text):
    assert isinstance(text, pd.core.series.Series)
    return text.str.split("\n").apply(lambda x: x[1:-2])


def preprocess_spec(data):
    
    assert isinstance(data, pd.core.frame.DataFrame)
    
    data["스펙1"] = split_spec(data["스펙"]) 
    
    data["학력"] = data["스펙1"].apply(lambda x: x[0])

    # 고졸인 사람들을 제외
    data["전공"] = data["스펙1"][data["스펙1"].apply(lambda x: len(x) > 1)].apply(lambda x: x[1])
    
    data["전공"] = re_sub(data["전공"], patterns={
        "자격증\s*\w*": " ",
        "제2외국어\s*\w*": " ",
        "토익\s*\w*": " ",
        "토스\s*\w*": " ",
        "오픽\s*\w*": " ",
        "인턴\s*\w*": " ",
        "-": " "
    })
    
    # 고졸인 사람들을 제외
    data["학점"] = data["스펙1"][data["스펙1"].apply(lambda x: len(x)) > 2].apply(lambda x: x[2])
    data["학점"] = re_sub(data["학점"], patterns={
        "자원봉사\w*\s*\d*": " ",
        "자격증\w*\s*\d*": " ",
        "해외경험\w*\s*\d*": " ",
        "수상\w*\s*\d*": " ",
        "\w*\s*회": " ",
        "\w*\s*개": " ",
        "인턴\w*\s*\d*": " ",
        "동아리\w*\s*\d*": " ",
        "토익\w*\s*\d*": " ",
        "-": " "
    })
    data["학점"] = data["학점"][data["학점"].notna()].str.split(" ").apply(lambda x: x[1])

    data["스펙2"] = re_sub(data["스펙"], patterns={
        "[\s]": " ",
        "\s*\d*읽음": " ",
        ",": " "
    })   

    for column in ["토익", "토스", "오픽", "사회활동"]:
        data[f"{column}"] = 0
        for i in data:
            data[f"{column}"] = data.스펙2.str.extract(f"({column}\s*\w*)")
            data[f"{column}"] = data[f"{column}"][data[f"{column}"].notna()].str.split(" ").apply(lambda x: x[1])
            
    for column in ["해외경험", "인턴", "수상", "동아리", "교내활동", "자원봉사"]:
        data[f"{column}"] = 0
        for i in data:
            data[f"{column}"] = data.스펙2.str.extract(f"({column}\s*\w*)")
            data[f"{column}"] = data[f"{column}"].str.extract("(\w*회)")

    data["자격증"] = 0
    for i in data:
        data["자격증"] = data.스펙2.str.extract("(자격증\s*\w*)")
        data["자격증"] = data.자격증.str.extract("(\w*개)")

    data.drop(columns=["스펙1", "스펙2"], inplace=True)