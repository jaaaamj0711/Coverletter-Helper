import re
import numpy as np
import pandas as pd

# str.replace 함수 정의 (특정 패턴 공백으로 처리)
def str_replace(text, patterns: list, replace=" "):
        
        assert isinstance(text, pd.core.series.Series)
        assert isinstance(patterns, list)

        for pattern in patterns:
                text = text.str.replace(pattern, replace)

        return text

# re.sub 함수 정의 (특정 정규식 패턴 공백으로 처리)
def re_sub(text, patterns: dict):
        
        assert isinstance(text, pd.core.series.Series)
        assert isinstance(patterns, dict)

        for pattern in patterns:
                text[text.notna()] = text[text.notna()].apply(lambda x: re.sub(pattern, patterns[pattern], x))

        return text

# 텍스트 최종 체크 함수
def filt(text, only_hangul=False, replace=" "):
    
    assert isinstance(text, pd.core.series.Series)

    pattern = "[^ㄱ-ㅎㅏ-ㅣ가-힣]" if only_hangul else "[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9&]"
    
    text[text.notna()] = text[text.notna()].apply(lambda x: re.sub(pattern, replace, x))
    text[text.notna()] = text[text.notna()].apply(lambda x: re.sub("\s+", replace, x))
    
    return text.str.strip()


# 회사명 전처리
def preprocess_company(data):
    
    assert isinstance(data, pd.core.frame.DataFrame)
    
    data["회사명"] = data["회사명"].str.lower()
    data["회사명"] = str_replace(data["회사명"], patterns=["㈜", "주식회사"]) 
    data["회사명"] = re_sub(data["회사명"], patterns={r"\([^)]*\)": " " })
    data["회사명"] = filt(data["회사명"])
    data.loc[data["회사명"].str[0] == "셰", "회사명"] = "셰플러코리아"
    
    return data


# 소제목 추출
def title(data):
    
    assert isinstance(data, pd.core.frame.DataFrame)
    
    data = data.sort_index()
    data2 = data.copy()
    data2["답변"] = re_sub(data2["답변"], patterns={ 
        "\r\n\"": "{",    
        "\"\r": "}",
        "\s*\[": "{",
        "\]": "}",
        "[\s]": " "
    })
    
    title_yes = data2[data2["답변"].str.contains('{' and '}')]
    index = title_yes.index
    title_no = data2.drop(index)
    
    # 소제목이 없는 부분
    title_no["제목"] = np.nan
    
    # 소제목이 있는 부분
    title_yes["제목"] = re_sub(title_yes["답변"], patterns={
        "\{": "",
        "\}.*": ""
    })
    
    # 데이터 통합
    title = pd.concat([title_yes["제목"], title_no["제목"]], axis=0).sort_index()
    data["소제목"] = title
    
    return data

# 자기소개서 답변 전처리 함수
def preprocess_answer(data):
    
    assert isinstance(data, pd.core.frame.DataFrame)
    
    data = title(data)
    data["답변"] = data["답변"].str.lower()
    data["답변"] = re_sub(data["답변"], patterns={
        "아쉬운점\s\d": " ",
        "좋은점\s\d": " ",
        "글자수\s\d{1,}[,]?\d{1,}자\d{1,}[,]?\d{1,}byte": " ",
        "[.]{1,}": " ",
        "[,]{1,}": " ",
        "o{3,}": " ",
        "[\s]": " "
    })
    data["답변"] = filt(data["답변"])
    
    return data

# 직무분야 분리 함수
def preprocess_job(data):
    
    data["직무분야"] = data["직무분야"].str.split("·") 
    
    for i in range(data["직무분야"].apply(lambda x: len(x)).max()):
        index = data["직무분야"].apply(lambda x: len(x) >= (i+1))
        data.loc[index, f"직무분야{i+1}"] = data.loc[index, "직무분야"].apply(lambda x: x[i])
    
    data.drop(columns=["직무분야"], inplace=True)
    
    return data
