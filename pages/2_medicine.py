# streamlit_app.py
import streamlit as st
import pandas as pd
from konlpy.tag import Okt

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
    return result[['itemName_clean', 'entpName', 'efcyQesitm', 'atpnQesitm', 'atpnWarnQesitm', 'seQesitm', 'score']]

# 데이터 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("pages/medicine_info_with_nouns.csv")
    df = df.fillna("")
    return df

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
                st.markdown(f"### {row['itemName_clean']} ({row['entpName']})")
                st.markdown(f"**✔️ 주요 효능:** {row['efcyQesitm'][:100]}{'...' if len(row['efcyQesitm']) > 100 else ''}")
                st.markdown(f"**🔗 공통 키워드 개수:** `{row['score']}`")
                
                with st.expander("🔍 상세 보기"):
                    st.markdown(f"**📌 전체 효능 설명**\n\n{row['efcyQesitm']}")
                    st.markdown(f"**⚠️ 주의사항**\n\n{row.get('atpnQesitm', '정보 없음')}")
                    st.markdown(f"**⚠️ 주의사항 경고**\n\n{row.get('atpnWarnQesitm', '정보 없음')}")
                    st.markdown(f"**🚫 부작용**\n\n{row.get('seQesitm', '정보 없음')}")
                
                st.markdown("---")
    else:
        st.warning("😥 관련된 의약품을 찾을 수 없습니다. 다른 증상을 입력해보세요.")
