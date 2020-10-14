import re
import numpy as np
import pandas as pd

# str.replace 함수 정의 (특정 패턴 공백으로 처리)
def str_replace(text, patterns: list, replace=" "):
        
        assert isinstance(text,pd.core.series.Series)
        assert isinstance(patterns,list)

        for pattern in patterns:
                text = text.str.replace(pattern, replace)

        return text

# re.sub 함수 정의 (특정 정규식 패턴 공백으로 처리)
def re_sub(text, patterns: dict):
        
        assert isinstance(text,pd.core.series.Series)
        assert isinstance(patterns,dict)

        for pattern in patterns:
                text[text.notna()] = text[text.notna()].apply(lambda x: re.sub(pattern, patterns[pattern], x))

        return text

def filt(text, only_hangul=False, replace=" "):
    
    assert isinstance(text, pd.core.series.Series)

    pattern = "[^ㄱ-ㅎㅏ-ㅣ가-힣]" if only_hangul else "[^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z0-9&]"
    
    text[text.notna()] = text[text.notna()].apply(lambda x: re.sub(pattern, replace, x))
    text[text.notna()] = text[text.notna()].apply(lambda x: re.sub("\s+", replace, x))
    
    return text.str.strip()
~   
