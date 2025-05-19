import requests
import urllib3
import json
import os
import re
from dotenv import load_dotenv

load_dotenv()

accessKey = os.getenv("ETRI_API_KEY")

# 사용할 API 주소 (문어체)
openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"

# 사용자로부터 문장 입력
text = input("문장을 입력하세요: ")

# 접속 어미 기준으로 문장 절 나누기 함수 (빈 절 제거 포함)
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
        if part:  # 빈 절 제거
            clauses.append(part)
    return clauses

# 부정어 목록
neg_words = ['안', '못', '별로', '전혀', '아니', '아닌', '없', '않', '괜찮']

# 절 긍정/부정 분류
def classify_clauses(clauses):
    positive = []
    negative = []
    for clause in clauses:
        if any(neg in clause for neg in neg_words):
            negative.append(clause)
        else:
            positive.append(clause)
    return positive, negative

# ETRI API 호출 함수 (형태소 분석)
def call_etri_api(text):
    analysisCode = "morp"
    requestJson = {
        "access_key": accessKey,
        "argument": {
            "text": text,
            "analysis_code": analysisCode
        }
    }

    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        openApiURL,
        headers={
            "Content-Type": "application/json; charset=UTF-8",
            "Authorization": accessKey
        },
        body=json.dumps(requestJson)
    )

    if response.status != 200:
        print(f"API 호출 실패: {response.status}")
        return None

    return json.loads(response.data.decode("utf-8"))

# 주어, 동사 추출 함수
def extract_subject_verb(morp_json):
    # 형태소 분석 결과에서 문장 단위로 형태소 리스트 가져오기
    subjects = []
    verbs = []

    sentences = morp_json.get('return_object', {}).get('sentence', [])
    for sentence in sentences:
        morp_list = sentence.get('morp', [])
        
        # 형태소 리스트에 'lemma'와 'type' 필드 존재
        for idx, morph in enumerate(morp_list):
            ## 주어 추출: 명사(NNG, NNP, NNB) + 조사(JKS, JX, JKG, JKV, JKQ)
            if morph['type'] in ['NNG', 'NNP', 'NNB']:
                next_idx = idx + 1
                if next_idx < len(morp_list):
                    next_morph = morp_list[next_idx]
                    if next_morph['type'] in ['JKS', 'JX', 'JKG', 'JKV', 'JKQ']:
                        subj = morph['lemma'] + next_morph['lemma']
                        subjects.append(subj)
            # 동사 추출: 용언(동사, 형용사 등)
            if morph['type'] in ['VV', 'VA']:
                verbs.append(morph['lemma'])

    return list(set(subjects)), list(set(verbs))  # 중복 제거

# 메인 실행

clauses = split_sentences_by_conj_endings(text)
pos_clauses, neg_clauses = classify_clauses(clauses)

print("✅ 긍정 절:", pos_clauses)
print("❌ 부정 절:", neg_clauses)

print("\n[긍정 절 분석]")
for i, clause in enumerate(pos_clauses, 1):
    print(f"{i}. \"{clause}\"")
    api_result = call_etri_api(clause)
    if api_result:
        subjs, vers = extract_subject_verb(api_result)
        print("주어:", subjs if subjs else "없음")
        print("동사:", vers if vers else "없음")
    else:
        print("API 호출 실패로 분석 불가")
    print()

print("[부정 절 분석]")
for i, clause in enumerate(neg_clauses, 1):
    print(f"{i}. \"{clause}\"")
    api_result = call_etri_api(clause)
    if api_result:
        subjs, vers = extract_subject_verb(api_result)
        print("주어:", subjs if subjs else "없음")
        print("동사:", vers if vers else "없음")
    else:
        print("API 호출 실패로 분석 불가")
    print()
