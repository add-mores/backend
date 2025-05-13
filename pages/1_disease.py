# pages/1_disease.py
import streamlit as st
import os
import sys
import pandas as pd
import psycopg2
from dotenv import load_dotenv

st.set_page_config(page_title="질병 정보", layout="wide")

# code/streamlit 디렉토리 경로를 Python 경로에 추가
utils_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'code', 'streamlit')
sys.path.insert(0, utils_dir)

# 이제 _utils.py를 직접 import 가능
from _utils import (
    process_tokens, calculate_tfidf_weights, calculate_tfidf_similarity, apply_symptom_bonus
)

# 환경 변수 로드
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

# 제목과 설명
st.title("🏥 질병 정보 검색 (TF-IDF 적용)")
st.markdown("증상이나 질병명을 입력하여 관련 질병 정보를 찾아보세요.")

# CSS 스타일 추가 (다크 모드 호환성 개선)
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
    .token-chip {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.2) !important;
        border-radius: 16px;
        padding: 2px 8px;
        margin: 2px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# DB 연결 함수
def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

# 사이드바 - 검색 옵션
st.sidebar.header("검색 옵션")
search_method = st.sidebar.radio(
    "검색 방식",
    ["TF-IDF 기반 검색", "단순 매칭 검색"]
)

# 검색 UI
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
            # 사용자 입력 토큰화
            user_tokens = user_input.lower().split()
            
            # DB 연결 및 질병 검색
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # 모든 질병 데이터 가져오기
            cursor.execute("""
                SELECT disnm_ko, disnm_en, dep, definition, symptoms, tokens
                FROM testdis
            """)
            
            all_diseases = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # 선택한 검색 방법에 따라 처리
            if search_method == "TF-IDF 기반 검색":
                # TF-IDF 가중치 계산 (실제로는 한 번만 계산하고 캐싱하는 것이 효율적)
                disease_weights = calculate_tfidf_weights(all_diseases)
                
                # TF-IDF 유사도 계산
                disease_scores = calculate_tfidf_similarity(user_tokens, disease_weights)
                
                # 증상 필드 보너스 적용
                disease_scores = apply_symptom_bonus(user_tokens, disease_scores, all_diseases)
                
                # 결과 정렬 및 변환
                top_results = []
                for disease_id, score in sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)[:10]:
                    # 해당 질병 정보 찾기
                    disease = next((d for d in all_diseases if d[0] == disease_id), None)
                    if disease:
                        top_results.append((disease, score))
                
            else:  # 단순 매칭 검색
                # 기존 방식 (단순 토큰 매칭)
                disease_scores = []
                
                for disease in all_diseases:
                    disnm_ko, disnm_en, dep, definition, symptoms, tokens = disease
                    
                    # 토큰 처리
                    disease_tokens = process_tokens(tokens)
                    
                    # 유사도 계산 (기존 방식)
                    from utils import calculate_similarity
                    score = calculate_similarity(user_tokens, disease_tokens, symptoms)
                    
                    # 질병명에 키워드가 직접 포함되면 가중치 추가
                    for token in user_tokens:
                        if disnm_ko and token in disnm_ko.lower():
                            score += 3
                    
                    if score > 0:
                        disease_scores.append((disease, score))
                
                # 결과 정렬
                top_results = sorted(disease_scores, key=lambda x: x[1], reverse=True)[:10]
            
            # 결과 표시
            if top_results:
                st.success(f"{len(top_results)}개의 관련 질병을 찾았습니다.")
                
                # 사용자 입력 토큰 표시
                st.markdown("### 입력 키워드")
                tokens_html = ""
                for token in user_tokens:
                    tokens_html += f'<div class="token-chip">{token}</div>'
                st.markdown(f"<div>{tokens_html}</div>", unsafe_allow_html=True)
                
                # 질병 결과 표시
                st.markdown("### 관련 질병")
                
                for i, (disease, score) in enumerate(top_results):
                    disnm_ko, disnm_en, dep, definition, symptoms, tokens = disease
                    disease_tokens = process_tokens(tokens)
                    
                    # 상위 토큰과 가중치 (TF-IDF 방식인 경우)
                    if search_method == "TF-IDF 기반 검색" and disnm_ko in disease_weights:
                        token_weights = disease_weights[disnm_ko]
                        top_tokens = sorted(token_weights.items(), key=lambda x: x[1], reverse=True)[:15]
                    else:
                        # 단순 매칭 방식인 경우 토큰만 표시
                        top_tokens = [(token, 1.0) for token in disease_tokens[:15]]
                    
                    with st.expander(f"{i+1}. {disnm_ko} ({disnm_en})", expanded=(i==0)):
                        st.markdown(f"<div class='score-badge'>관련도: {score:.2f}</div>", unsafe_allow_html=True)
                        
                        # 진료과
                        st.markdown("<p class='category-label'>📋 진료과</p>", unsafe_allow_html=True)
                        st.markdown(f"<div class='content-box'>{dep if dep else '정보 없음'}</div>", unsafe_allow_html=True)
                        
                        # 정의
                        st.markdown("<p class='category-label'>📝 정의</p>", unsafe_allow_html=True)
                        st.markdown(f"<div class='content-box'>{definition if definition else '정보 없음'}</div>", unsafe_allow_html=True)
                        
                        # 증상
                        st.markdown("<p class='category-label'>🔍 주요 증상</p>", unsafe_allow_html=True)
                        st.markdown(f"<div class='content-box'>{symptoms if symptoms else '정보 없음'}</div>", unsafe_allow_html=True)
                        
                        # 관련 키워드
                        st.markdown("<p class='category-label'>🔑 관련 키워드</p>", unsafe_allow_html=True)
                        tokens_html = ""
                        for token, weight in top_tokens:
                            tokens_html += f'<div class="token-chip">{token} ({weight:.2f})</div>'
                        st.markdown(f"<div class='content-box'>{tokens_html}</div>", unsafe_allow_html=True)
            else:
                st.error("검색 결과가 없습니다. 다른 키워드로 시도해보세요.")

# 페이지 이동 버튼
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    if st.button("👨‍⚕️ 관련 의약품 보기", use_container_width=True):
        # 현재 결과를 세션에 저장
        if 'top_results' in locals():
            st.session_state.disease_results = top_results
        st.switch_page("pages/3_medicine.py")
with col2:
    if st.button("🗺️ 관련 병원 찾기", use_container_width=True):
        # 현재 결과를 세션에 저장
        if 'top_results' in locals():
            st.session_state.disease_results = top_results
        st.switch_page("pages/2_hospital.py")