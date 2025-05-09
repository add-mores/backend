import streamlit as st
import pandas as pd
import os
import psycopg2
from dotenv import load_dotenv
from konlpy.tag import Okt

# 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(page_title="의약품 추천", layout="wide")

# 형태소 분석기
okt = Okt()


# 사용자 입력에서 명사 추출
def extract_nouns(text):
    return [n for n in okt.nouns(text) if len(n) > 1]

# 추천 함수 (10점 만점 기반 가중치 적용)
def recommend_with_weights(user_input, df, top_n=5):
    user_nouns = set(extract_nouns(user_input))
    user_noun_count = len(user_nouns)
    # 약물 복용 시 부작용 가능성이 높은 고위험군이나 장기 기능 저하 상태를 나타내는 키워드로, 해당 내용이 없을수록 대부분의 사람이 안전하게 복용할 수 있음
    RISK_KEYWORDS = ['과민증', '어린이', '고령자', '간장애', '신장애', '임산부', '수유부', '간질환', '신부전']

    if user_noun_count == 0:
        return pd.DataFrame(columns=[
            'itemname_clean', 'entpname', 'efcyqesitm',
            'atpnqesitm', 'atpnwarnqesitm', 'seqesitm', 'total_score'
        ])

    df = df.fillna('')

    # 효능 관련도 점수: 입력 증상 중 겹치는 비율 * 6점
    def symptom_score(efcy_nouns):
        med_nouns = set(efcy_nouns.split(","))
        overlap = len(user_nouns & med_nouns)
        return (overlap / user_noun_count) * 6

    # 주의사항 경고 점수: 없으면 1점
    def warn_score(text):
        return 1 if not text.strip() else 0

    # 주의사항 위험 키워드 점수: 적게 포함될수록 높음 (0~2점)
    def caution_score(text):
        count = sum(1 for word in RISK_KEYWORDS if word in text)
        return max(0, 4 - count) / 2  # 0~2점

    # 부작용 설명 점수: 짧을수록 좋음 (0~1점)
    avg_len = 196.22  # 데이터 기준 평균
    def side_effect_score(text):
        length = len(text)
        return max(0, 1 - (length / (avg_len * 2)))  # 평균의 2배 이상이면 0점

    # 점수 계산
    df['symptom_score'] = df['efcy_nouns'].apply(symptom_score)
    df['warn_score'] = df['atpnwarnqesitm'].apply(warn_score)
    df['caution_score'] = df['atpnqesitm'].apply(caution_score)
    df['side_effect_score'] = df['seqesitm'].apply(side_effect_score)

    df['total_score'] = (
        df['symptom_score'] +
        df['warn_score'] +
        df['caution_score'] +
        df['side_effect_score']
    )

    # 소수점 두 자리까지
    df['total_score'] = df['total_score'].round(2)

    result = df[df['symptom_score'] > 0].sort_values(by='total_score', ascending=False).head(top_n)

    return result[[
        'itemname_clean', 'entpname', 'efcyqesitm',
        'atpnqesitm', 'atpnwarnqesitm', 'seqesitm', 'total_score'
    ]]


# DB 연결 함수
def get_db_connection():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        st.error("DATABASE_URL 환경 변수가 설정되지 않았습니다.")
        return None
    try:
        return psycopg2.connect(DATABASE_URL)
    except Exception as e:
        st.error(f"데이터베이스 연결 오류: {e}")
        return None

# DB에서 데이터 불러오기
@st.cache_data
def load_data():
    conn = get_db_connection()
    if conn is None:
        return pd.DataFrame()  # 빈 DataFrame 반환

    try:
        query = "SELECT * FROM testmed"
        df = pd.read_sql(query, conn)
        df = df.fillna("")
        return df
    except Exception as e:
        st.error(f"데이터 조회 오류: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# Streamlit 앱 구성
st.title("💊 의약품 추천")
st.write("입력한 증상과 가장 관련된 의약품을 추천합니다.")

user_input = st.text_input("📝 증상 또는 질환 입력", placeholder="예: 소화불량, 기침, 위염")

if user_input:
    df = load_data()
    result = recommend_with_weights(user_input, df)

    if not result.empty:
        st.subheader("📋 추천 의약품 목록")

        for i, row in result.iterrows():
            with st.container():
                st.markdown(f"### {row['itemname_clean']} ({row['entpname']})")
                st.markdown(f"**✔️ 주요 효능:** {row['efcyqesitm'][:100]}{'...' if len(row['efcyqesitm']) > 100 else ''}")
                st.markdown(f"**🔗 관련도:** `{row['total_score']}`")
                
                with st.expander("🔍 상세 보기"):
                    st.markdown(f"**📌 전체 효능 설명**\n\n{row['efcyqesitm']}")
                    st.markdown(f"**⚠️ 주의사항**\n\n{row.get('atpnqesitm', '정보 없음')}")
                    st.markdown(f"**⚠️ 주의사항 경고**\n\n{row.get('atpnwarnqesitm', '정보 없음')}")
                    st.markdown(f"**🚫 부작용**\n\n{row.get('seqesitm', '정보 없음')}")
                
                st.markdown("---")
    else:
        st.warning("😥 관련된 의약품을 찾을 수 없습니다. 다른 증상을 입력해보세요.")
