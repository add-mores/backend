import pandas as pd
import re
from konlpy.tag import Okt

# CSV 불러오기
df = pd.read_csv("~/temp/add-more/data/medicine_info_cleaned.csv")
df['efcyQesitm'] = df['efcyQesitm'].fillna("")

# 분석기 초기화
okt = Okt()

# 질환 관련 키워드 (시작어 기준)
start_keywords = [
    "병", "질병", "증후군", "염", "증", "통", "불량", "급성", "기능", "장애", "궤양",
    "헤르니아", "치료", "백선", "성", "세", "균", "통증", "불균형", "마비", "암", "결핵", "팽만감", "팽만", "보조제"
]

# 제외할 일반 단어
stopwords = {"사용"}

def clean_text_and_get_parens(text):
    # 괄호 안 단어 추출
    paren_raw = re.findall(r'\(([^)]+)\)', text)
    paren_words = []
    
    # 괄호 속 내용에서 명사만 추출
    for p in paren_raw:
        paren_words += [n for n in okt.nouns(p) if len(n) > 1 and n not in stopwords]
    
    # 괄호 제거된 텍스트
    cleaned_text = re.sub(r'\([^)]*\)', '', text)
    return cleaned_text, paren_words

def extract_terms_with_subwords(text):
    cleaned_text, paren_words = clean_text_and_get_parens(text)
    raw_words = re.findall(r'[가-힣]{2,}', cleaned_text)

    # 키워드로 시작하는 단어는 전체 단어로 추출
    keyword_terms = []
    for word in raw_words:
        for kw in start_keywords:
            if word.startswith(kw):  # 키워드로 시작하는 단어는 전체 유지
                keyword_terms.append(word)
                break

    # 키워드가 포함된 단어에서 키워드 뒤를 자르는 로직
    general_nouns = []
    for word in raw_words:
        for kw in start_keywords:
            if kw in word and not word.startswith(kw):  # 키워드가 포함되었지만 시작하지 않는 경우
                general_nouns.append(word.split(kw)[0] + kw)
                break
    
    # 형태소 분석 (명사 추출)
    nouns = okt.nouns(cleaned_text)
    final_nouns = [n for n in nouns if len(n) > 1 and n not in stopwords]
    
    # 통합 후 중복 제거
    all_terms = keyword_terms + general_nouns + final_nouns + paren_words
    return ",".join(sorted(set(all_terms)))

# 적용
df['efcy_nouns'] = df['efcyQesitm'].apply(extract_terms_with_subwords)

# 저장
df.to_csv("~/temp/add-more/data/medicine_info_with_nouns.csv", index=False)
print("질환 키워드와 괄호 단어 포함 형태소 추출 완료")
