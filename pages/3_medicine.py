import streamlit as st
import pandas as pd
import os
import psycopg2
from dotenv import load_dotenv
from konlpy.tag import Okt
import math
from collections import Counter
from sqlalchemy import create_engine

# 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(page_title="의약품 추천", layout="wide")

# 형태소 분석기
okt = Okt()

# 사용자 입력에서 명사 추출
def extract_nouns(text):
    return [n for n in okt.nouns(text) if len(n) > 1]

# DB 연결 함수
def get_engine():
    # 환경변수에서 DB 주소 가져오기
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        st.error("DATABASE_URL 환경 변수가 없어요!")
        return None
    return create_engine(db_url)

# 데이터 불러오기 함수
@st.cache_data
def load_data():
    engine = get_engine()
    if engine is None:
        return pd.DataFrame()
    query = "SELECT * FROM testmed"
    return pd.read_sql(query, engine)

# efcy_nouns 컬럼에서 IDF 점수 계산
def compute_idf_scores(df):
    doc_count = len(df)
    noun_doc_freq = Counter()

    for entry in df['ph_effect_c'].dropna():
        nouns = set(n.strip() for n in entry.split(',') if n.strip())
        for noun in nouns:
            noun_doc_freq[noun] += 1

    idf_scores = {
        noun: math.log(doc_count / (1 + freq))  # 1을 더해 0 division 방지
        for noun, freq in noun_doc_freq.items()
    }

    return idf_scores

# 추천 함수 (10점 만점 기반 가중치 적용 + 사용자 조건 필터링)
def recommend_with_weights(user_input, df, age_group=None, is_pregnant=False, has_disease=None, top_n=5):
    user_nouns = set(extract_nouns(user_input))
    user_noun_count = len(user_nouns)
    RISK_KEYWORDS = ['과민증', '어린이', '고령자', '간장애', '신장애', '임산부', '수유부', '간질환', '신부전']

    if user_noun_count == 0:
        return pd.DataFrame(columns=[
            'ph_nm_c', 'ph_c_nm', 'ph_effect',
            'ph_anti_warn', 'ph_warn', 'ph_s_effect', 'total_score'
        ])

    df = df.fillna("")
    # IDF 점수 계산
    idf_scores = compute_idf_scores(df)

    # 사용자 조건 필터링 함수
    def exclude_by_user_conditions(row):
        full_text = f"{row['ph_effect']} {row['ph_anti_warn']} {row['ph_warn']}"

        # 연령 필터
        if age_group:
            age_keywords = {
                '소아': ['소아', '어린이', '유아', '영아', '아동'],
                '청소년': ['청소년', '10대', '10세', '십대'],
                '노인': ['노인', '고령자'],
                '성인': ['성인']
            }
            for keyword in age_keywords.get(age_group, [age_group]):
                if keyword in full_text:
                    return False

        # 임신 필터
        if is_pregnant:
            if any(keyword in full_text for keyword in ['임산부', '임신', '임부']):
                return False

        # 질환 필터 (입력 기반)
        if has_disease:
            for disease in has_disease:
                if disease in full_text:
                    return False

        return True
    
    df = df[df.apply(exclude_by_user_conditions, axis=1)]

    # IDF 기반 점수 계산
    def symptom_score(efcy_nouns):
        med_nouns = set(n.strip() for n in efcy_nouns.split(",") if n.strip())
        overlap_nouns = user_nouns & med_nouns
        if not overlap_nouns:
            return 0
        score = sum(idf_scores.get(noun, 0) for noun in overlap_nouns)
        return min(score, 6)  # 점수 상한

    def warn_score(text):
        return 1 if not text.strip() else 0

    def caution_score(text):
        count = sum(1 for word in RISK_KEYWORDS if word in text)
        return max(0, 4 - count) / 2

    avg_len = 196.22
    def side_effect_score(text):
        length = len(text)
        return max(0, 1 - (length / (avg_len * 2)))

    # 점수 적용
    df['symptom_score'] = df['ph_effect_c'].apply(symptom_score)
    df['warn_score'] = df['ph_warn'].apply(warn_score)
    df['caution_score'] = df['ph_anti_warn'].apply(caution_score)
    df['side_effect_score'] = df['ph_s_effect'].apply(side_effect_score)

    df['total_score'] = (
        df['symptom_score'] +
        df['warn_score'] +
        df['caution_score'] +
        df['side_effect_score']
    ).round(2)

    result = df[df['symptom_score'] > 0].sort_values(by='total_score', ascending=False).head(top_n)

    return result[[  # 최종 출력 열
        'ph_nm_c', 'ph_c_nm', 'ph_effect',
        'ph_anti_warn', 'ph_warn', 'ph_s_effect', 'total_score'
    ]]

# ------------------------- Streamlit UI -------------------------

st.title("💊 의약품 추천")
st.write("입력한 증상과 가장 관련된 의약품을 추천합니다.")

# 사용자 입력
user_input = st.text_input("📝 증상 또는 질환 입력", placeholder="예: 소화불량, 기침, 위염")


# 사용자 조건
with st.expander("사용자 조건 선택"):
    age_group = st.selectbox("연령대 선택", ["", "소아", "청소년", "성인", "고령자"])
    is_pregnant = st.checkbox("임신 중")
    disease_input = st.text_input("🏥 피하고 싶은 질병명 (쉼표로 구분)", placeholder="예: 간질환, 신장병")
    if disease_input.strip():
        has_disease = [d.strip() for d in disease_input.split(',') if d.strip()]
    else:
        has_disease = []

# 추천 기준 안내
with st.expander("추천 기준 안내"):
    st.markdown("""
    - ✔️ **증상 관련도**: 입력 증상과 효능이 얼마나 겹치는지  
    - ⚠️ **주의사항 경고 없음**: 경고 문구가 없으면 가산점  
    - 🔍 **위험 키워드 적음**: 과민증, 임산부, 신장애 등 키워드 적을수록 점수 상승  
    - 🚫 **부작용 설명 짧음**: 부작용 항목이 짧을수록 선호  
    - 🧍 **사용자 조건 필터링**: 연령, 임신, 질환 관련 키워드가 있는 약은 제외  
    """)

# 검색 버튼
if st.button("🔍 의약품 검색") and user_input:
    df = load_data()
    result = recommend_with_weights(user_input, df, age_group, is_pregnant, has_disease)

    if not result.empty:
        st.subheader("📋 추천 의약품 목록")
        for _, row in result.iterrows():
            with st.container():
                st.markdown(f"### {row['ph_nm_c']} ({row['ph_c_nm']})")
                st.markdown(f"**✔️ 주요 효능:** {row['ph_effect'][:100]}{'...' if len(row['ph_effect']) > 100 else ''}")
                st.markdown(f"**🔗 관련도:** `{row['total_score']}`")
                with st.expander("🔍 상세 보기"):
                    st.markdown(f"**📌 전체 효능 설명**\n\n{row['ph_effect']}")
                    st.markdown(f"**⚠️ 주의사항**\n\n{row['ph_anti_warn'] or '정보 없음'}")
                    st.markdown(f"**⚠️ 주의사항 경고**\n\n{row['ph_warn'] or '정보 없음'}")
                    st.markdown(f"**🚫 부작용**\n\n{row['ph_s_effect'] or '정보 없음'}")
                st.markdown("---")
    else:
        st.warning("😥 관련된 의약품을 찾을 수 없습니다. 조건을 변경하거나 다른 증상을 입력해보세요.")
