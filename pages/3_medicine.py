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

# 추천 함수
def recommend_by_overlap(user_input, df, top_n=5):
    user_nouns = set(extract_nouns(user_input))
    
    def overlap_score(efcy_nouns):
        med_nouns = set(efcy_nouns.split(","))
        return len(user_nouns & med_nouns)
    
    df['score'] = df['efcy_nouns'].apply(overlap_score)
    result = df[df['score'] > 0].sort_values(by='score', ascending=False).head(top_n)
    return result[['itemname_clean', 'entpname', 'efcyqesitm', 'atpnqesitm', 'atpnwarnqesitm', 'seqesitm', 'score']]

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
    result = recommend_by_overlap(user_input, df)

    if not result.empty:
        st.subheader("📋 추천 의약품 목록")

        for i, row in result.iterrows():
            with st.container():
                st.markdown(f"### {row['itemname_clean']} ({row['entpname']})")
                st.markdown(f"**✔️ 주요 효능:** {row['efcyqesitm'][:100]}{'...' if len(row['efcyqesitm']) > 100 else ''}")
                st.markdown(f"**🔗 공통 키워드 개수:** `{row['score']}`")
                
                with st.expander("🔍 상세 보기"):
                    st.markdown(f"**📌 전체 효능 설명**\n\n{row['efcyqesitm']}")
                    st.markdown(f"**⚠️ 주의사항**\n\n{row.get('atpnqesitm', '정보 없음')}")
                    st.markdown(f"**⚠️ 주의사항 경고**\n\n{row.get('atpnwarnqesitm', '정보 없음')}")
                    st.markdown(f"**🚫 부작용**\n\n{row.get('seqesitm', '정보 없음')}")
                
                st.markdown("---")
    else:
        st.warning("😥 관련된 의약품을 찾을 수 없습니다. 다른 증상을 입력해보세요.")
