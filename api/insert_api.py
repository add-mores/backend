from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import re
import json
import urllib3
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Input 데이터 모델
class TextInput(BaseModel):
    text: str

# 접속 어미 기준으로 문장 절 나누기
def split_sentences_by_conj_endings(text):
    split_pattern = re.compile(r'(데|고|지만|으나|는데|요)')
    splits = split_pattern.split(text)

    clauses = []
    i = 0
    while i < len(splits):
        part = splits[i]
        if i + 1 < len(splits) and splits[i + 1] in ['데', '고', '지만', '으나', '는데', '요']:
            part += splits[i + 1]
            i += 2
        else:
            i += 1
        part = part.strip()
        if part:
            clauses.append(part)
    return clauses

# 부정어 목록
neg_words = ['안', '못', '별로', '전혀', '아니', '아닌', '없', '않', '괜찮', '나아', '좋아']

# 절 긍정/부정 분류
def classify_clauses(clauses):
    positive = []
    negative = []
    # 정규표현식 기반 예외 패턴 (지 않고, 지않고 모두 포함)
    exception_patterns = [
        r"멈추지\s?않고",
        r"사라지지\s?않고",
        r"좋아지지\s?않고",
        r"나아지지\s?않고"
    ]

    for clause in clauses:
        # 예외 패턴에 해당하는 경우
        if any(re.search(pattern, clause) for pattern in exception_patterns):
            positive.append(clause)
        # 부정어 포함된 경우
        elif any(neg in clause for neg in neg_words):
            negative.append(clause)
        else:
            positive.append(clause)

    return positive, negative


# FastAPI 라우트
@app.post("/insert")
def insert_text(data: TextInput):
    text = data.text
    clauses = split_sentences_by_conj_endings(text)
    pos_clauses, neg_clauses = classify_clauses(clauses)

    result = {
        "original_text": text,
        "positive": pos_clauses,
        "negative": neg_clauses
    }

    return result


# 루트 엔드포인트
@app.get("/")
def root():
    return {"message": "질병 추천을 위한 입력 API 입니다다."}