import streamlit as st
import os
import psycopg2
from dotenv import load_dotenv
from utils import process_tokens, calculate_similarity

# 환경 변수 로드
load_dotenv()

# 페이지 설정
st.set_page_config(page_title="질병 정보", layout="wide")

# 제목과 설명
st.title("🏥 질병 정보 검색")
st.markdown("증상이나 질병명을 입력하여 관련 질병 정보를 찾아보세요.")

# CSS 스타일 추가 - 다크 모드 호환성 개선
st.markdown("""
<style>
    .category-label {
        font-weight: bold;
        color: #4dabf7 !important;
        margin-bottom: 0px;
    }
    .content-box {
        background-color: rgba(70, 70, 70, 0.2) !important;
        color: inherit !important;
        border: 1px solid rgba(120, 120, 120, 0.5) !important;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .disease-title {
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
        background-color: rgba(59, 130, 246, 0.2) !important;
        border-radius: 5px;
        margin-bottom: 10px;
        color: inherit !important;
    }
    .score-badge {
        background-color: #4dabf7 !important;
        color: white !important;
        padding: 2px 6px;
        border-radius: 10px;
        font-size: 12px;
        float: right;
    }
</style>
""", unsafe_allow_html=True)

# DB 연결 함수
def get_db_connection():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        st.error("DATABASE_URL 환경 변수가 설정되지 않았습니다.")
        return None
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    except Exception as e:
        st.error(f"데이터베이스 연결 오류: {e}")
        return None

# 검색 UI 개선
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("질병명이나 증상을 입력하세요", placeholder="예: 두통, 감기, 발열...", label_visibility="collapsed")
with col2:
    search_button = st.button("검색", use_container_width=True)

if search_button or user_input:
    if not user_input:
        st.warning("검색어를 입력해주세요.")
    else:
        with st.spinner("검색 중..."):
            # DB 연결
            conn = get_db_connection()
            if conn:
                cursor = conn.cursor()
                
                # 모든 질병 데이터 가져오기
                cursor.execute("""
                    SELECT disnm_ko, disnm_en, dep, definition, symptoms, tokens
                    FROM testdis
                """)
                
                all_diseases = cursor.fetchall()
                cursor.close()
                conn.close()
                
                # 검색 키워드 토큰화
                search_tokens = user_input.lower().split()
                
                # 질병별 유사도 계산
                disease_scores = []
                
                for disease in all_diseases:
                    disnm_ko, disnm_en, dep, definition, symptoms, tokens = disease
                    
                    # 토큰 처리
                    disease_tokens = process_tokens(tokens)
                    
                    # 유사도 계산
                    score = calculate_similarity(search_tokens, disease_tokens, symptoms)
                    
                    # 질병명에 키워드가 직접 포함되면 가중치 추가
                    for token in search_tokens:
                        if disnm_ko and token in disnm_ko.lower():
                            score += 3
                    
                    if score > 0:
                        disease_scores.append((disease, score))
                
                # 결과 정렬 및 표시
                disease_scores.sort(key=lambda x: x[1], reverse=True)
                top_results = disease_scores[:10]
                
                if top_results:
                    st.success(f"{len(top_results)}개의 관련 질병을 찾았습니다.")
                    
                    for i, (disease, score) in enumerate(top_results):
                        disnm_ko, disnm_en, dep, definition, symptoms, _ = disease
                        
                        # 개선된 표시 방법
                        with st.expander(f"{i+1}. {disnm_ko} ({disnm_en})", expanded=(i==0)):
                            st.markdown(f"<div class='score-badge'>관련도: {score}</div>", unsafe_allow_html=True)
                            
                            # 진료과
                            st.markdown("<p class='category-label'>📋 진료과</p>", unsafe_allow_html=True)
                            st.markdown(f"<div class='content-box'>{dep if dep else '정보 없음'}</div>", unsafe_allow_html=True)
                            
                            # 정의
                            st.markdown("<p class='category-label'>📝 정의</p>", unsafe_allow_html=True)
                            st.markdown(f"<div class='content-box'>{definition if definition else '정보 없음'}</div>", unsafe_allow_html=True)
                            
                            # 증상
                            st.markdown("<p class='category-label'>🔍 주요 증상</p>", unsafe_allow_html=True)
                            st.markdown(f"<div class='content-box'>{symptoms if symptoms else '정보 없음'}</div>", unsafe_allow_html=True)
                else:
                    st.error("검색 결과가 없습니다. 다른 키워드로 시도해보세요.")