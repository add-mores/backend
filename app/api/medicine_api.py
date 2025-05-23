from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import List, Optional, Set
import pandas as pd
import os
import math
import json
from collections import Counter
from sqlalchemy import create_engine
from dotenv import load_dotenv
import urllib3

from app.api import insert_api

# 환경 변수 로드
load_dotenv()
router = APIRouter()

# 형태소 분석 API 설정
accessKey = os.getenv("ETRI_API_KEY")
openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"

# ETRI API 호출 함수
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
        headers={"Content-Type": "application/json; charset=UTF-8"},
        body=json.dumps(requestJson)
    )
    if response.status != 200:
        return None
    return json.loads(response.data.decode("utf-8"))


mapping_dict = {
    "근육통": [["아프", "뻐근하", "통증", "시큰거리", "시큰"]],
    "관절통": [["쑤시", "욱신거리", "통증", "시큰거리", "시큰"]],
    "신경통": [["저리", "찌릿", "통증", "시큰거리", "시큰"]],
    "두통, 편두통": [("머리", "아프"), ["지끈지끈", "지끈거리", "띵", "통증"]],
    "통증, 동통": [["아프", "쑤시", "찌릿", "따끔", "따끔거리", "욱신거리", "욱신", "통증", "시큰거리", "시큰"]],
    "요통": [("허리", "아프"), ["삐끗", "삐", "통증", "시큰거리", "시큰"]],
    "치통": [("이", "이빨"), ["시리", "아프", "욱신", "욱신거리", "찌릿", "쑤시", "통증", "시큰거리", "시큰"]],
    "위통": [("위", "아프"), ["속", "쓰리", "아프", "울렁거리", "통증"]],
    "복통": [("배", "아프"), ["꼬이", "통증"]],
    "골절통": [("뼈", "부러지"), ["아프"]],
    "병후, 수술후, 외상후": [["수술", "외상"], ["후", "하"]],
    "육체피로, 피로": [["피곤", "기운", "졸리", "처지", "나른하", "늘어지"]],
    "발열, 해열": [["열", "뜨겁", "끓"]],
    "어깨결림": [("어깨",), ["뻐근하", "뭉치", "굳", "결리"]],
    "오한": [["으슬으슬", "차갑", "차", "춥", "덜덜", "떨리"]],
    "코막힘": [("코", "콧"), ["막히"]],
    "입안염, 구각염, 입꼬리염, 입술염, 구내염": [("입", "입술"), ["헐", "따갑", "쓰", "트", "아프"]],
    "소염": [["염증"]],
    "구토, 구역": [["토", "게워냈", "뱉", "메스껍", "울렁거리", "니글거리", "느글거리"]],
    "식욕부진, 식욕감퇴": [("입맛", "밥", "식욕"), ["없", "떨어지"]],
    "인후통, 인후, 목구멍": [("목", "구멍"), ["칼칼하", "아프", "따갑"]],
    "수족냉증": [("손", "발"), ["차갑"]],
    "수족저림": [("손", "발"), ["저리"]],
    "위부팽만감, 위부불쾌감": [("배", "속", "위"), ["더부룩하", "부르", "빵빵", "불편", "불쾌", "답답"]],
    "부기": [["퉁퉁", "붓", "붓기"]],
    "목결림": [("목",), ["뻣뻣하", "움직이", "돌아가", "딱딱", "굳", "결리"]],
    "테니스엘보우": [("팔", "꿈치"), ["아프", "뻐근하", "시큰거리", "시큰", "저리", "찌릿"]],
    "어깨관절주위염": [("어깨",), ["아프", "뻐근하", "뻣뻣", "염증", "불편"]],
    "자상": [["찔리"]],
    "열상": [("피부", "칼"), ["베이"]],
    "좌상, 타박상": [["멍", "부딪히", "박"]],
    "복부": [["배"]],
    "건조감": [["건조", "뻑뻑"]],
    "완선": [("사타구니", "엉덩이"), ["가렵"]],
    "염좌, 염좌통": [["삐", "꺽이"]],
    "생리, 월경통, 생리통": [["생리", "통증", "월경"]],
    "빈혈": [["어지럽", "쓰러질거"]],
    "발치": [("이", "이빨"), ["뽑"]],
    "치은염": [("잇몸",), ["붓", "아프"]],
    "객담, 객담배출곤란": [["가래"]],
    "체함, 소화불량": [["체하", "소화"]],
    "위산과다, 신트림, 제산작용": [["공복", "속", "위"], ["쓰리", "아프"]],
}

# 형태소 리스트를 받아 mapping_dict 기반으로 효능 단어 리스트 반환
def match_effects(morp_list):
    morp_set = set(morp_list)
    effects_found = []

    for keys, effects in mapping_dict.items():
        # 'any' 옵션 없이 키워드 전부 있어야 매칭
        if all(k in morp_set for k in keys):
            effects_found.extend(effects)

    return effects_found

# 기존 extract_nouns_from_positive 대체
def extract_nouns_with_mapping(positive_clauses: List[str]) -> List[str]:
    all_nouns = []
    for clause in positive_clauses:
        json_data = call_etri_api(clause)
        if not json_data:
            continue

        # 형태소 추출
        morp_list = []
        sentences = json_data.get('return_object', {}).get('sentence', [])
        for sentence in sentences:
            morp_eval = sentence.get('morp_eval', [])
            for morp in morp_eval:
                # 형태소 원형 추가
                morp_list.append(morp.get('lemma', ''))

        # 매핑된 효능 단어 추가
        effects = match_effects(morp_list)
        morp_list.extend(effects)

        all_nouns.extend(morp_list)

    # 중복 제거 후 반환
    return list(set(all_nouns))


# DB 연결 함수
def get_engine():
    db_url = os.getenv("DATABASE_URL")
    return create_engine(db_url) if db_url else None

# IDF 점수 계산
def compute_idf_scores(df):
    doc_count = len(df)
    noun_doc_freq = Counter()
    for entry in df['ph_effect_c'].dropna():
        nouns = set(n.strip() for n in entry.split(',') if n.strip())
        for noun in nouns:
            noun_doc_freq[noun] += 1
    return {noun: math.log(doc_count / (1 + freq)) for noun, freq in noun_doc_freq.items()}

# 사용자 조건에 따른 필터링
def filter_by_conditions(df, age_group, is_pregnant, has_disease):
    def condition(row):
        full_text = f"{row['ph_effect']} {row['ph_anti_warn']} {row['ph_warn']}"
        age_keywords = {
            '소아': ['소아', '어린이', '유아', '영아', '아동'],
            '청소년': ['청소년', '10대', '10세', '십대'],
            '노인': ['노인', '고령자'],
            '성인': ['성인']
        }
        if age_group:
            for keyword in age_keywords.get(age_group, [age_group]):
                if keyword in full_text:
                    return False
        if is_pregnant and any(k in full_text for k in ['임산부', '임신', '임부']):
            return False
        if has_disease:
            for disease in has_disease:
                if disease in full_text:
                    return False
        return True
    return df[df.apply(condition, axis=1)]

# 점수 계산
RISK_KEYWORDS = ['과민증', '어린이', '고령자', '간장애', '신장애', '임산부', '수유부', '간질환', '신부전']

def score_row(row, user_nouns, idf_scores):
    med_nouns = set(n.strip() for n in row['ph_effect_c'].split(",") if n.strip())
    overlap_nouns = user_nouns & med_nouns
    symptom_score = sum(idf_scores.get(noun, 0) for noun in overlap_nouns)
    symptom_score = min(symptom_score, 6)
    warn_score = 1 if not row['ph_anti_warn'].strip() else 0
    caution_count = sum(1 for w in RISK_KEYWORDS if w in row['ph_warn'])
    caution_score = max(0, 4 - caution_count) / 2
    side_effect_score = max(0, 1 - len(row['ph_s_effect']) / (196.22 * 2))
    total_score = symptom_score + warn_score + caution_score + side_effect_score
    return round(total_score, 2)

# 요청 모델 정의
class RecommendRequest(BaseModel):
    symptom: str
    age_group: Optional[str] = None
    is_pregnant: Optional[bool] = False
    has_disease: Optional[List[str]] = []
    top_n: Optional[int] = 5

# 추천 API
@router.post("/api/medicine")
async def recommend_from_insert(request: Request):
    """
    insert API의 응답에서 'positive' 텍스트를 이용해 의약품 추천
    """
    body = await request.json()
    positive_clauses: List[str] = body.get("positive", [])
    
    # 고정값 또는 기본값 할당
    age_group = None       # 예: 기본값 없음
    is_pregnant = False   # 기본 임신 여부 False
    has_disease = []      # 기본 질병 없음


    if not positive_clauses:
        return {"error": "positive 문장이 없습니다."}

    # 긍정 문장 병합하여 증상 텍스트로 사용
    symptom_text = " ".join(positive_clauses)
    user_nouns = extract_nouns_with_mapping(positive_clauses)
    if not user_nouns:
        return {"result": []}


    engine = get_engine()
    if not engine:
        return {"error": "DB 연결 실패"}


    df = pd.read_sql("SELECT * FROM testmed", engine).fillna("")
    df = filter_by_conditions(df, age_group, is_pregnant, has_disease)
    idf_scores = compute_idf_scores(df)
    
    # 점수 계산산
    df['total_score'] = df.apply(lambda row: score_row(row, user_nouns, idf_scores), axis=1)
    df = df[df['total_score'] > 0].sort_values(by='total_score', ascending=False).head(5)

    return {
        "symptom_text": symptom_text,
        "result": df[[
            'ph_nm_c', 'ph_c_nm', 'ph_effect',
            'ph_anti_warn', 'ph_warn', 'ph_s_effect', 'total_score'
        ]].to_dict(orient="records")
    }
