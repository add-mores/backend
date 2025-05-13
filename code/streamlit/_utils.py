# utils.py
import json
import math
import numpy as np
from collections import Counter

def process_tokens(tokens):
    """토큰 데이터 처리"""
    if isinstance(tokens, str):
        try:
            return json.loads(tokens)
        except:
            return tokens.split()
    return tokens

def calculate_tfidf_weights(diseases):
    """질병 데이터에서 TF-IDF 가중치 계산"""
    # 모든 토큰 및 질병별 토큰 빈도 수집
    token_doc_count = {}  # 각 토큰이 나타나는 질병 수
    disease_tokens = {}  # 질병별 토큰 목록
    disease_token_freq = {}  # 질병별 토큰 빈도
    
    # 전처리 (토큰 추출 및 빈도 계산)
    for disease in diseases:
        disnm_ko, disnm_en, dep, definition, symptoms, tokens = disease
        
        # 토큰 처리
        tokens_list = process_tokens(tokens)
        disease_tokens[disnm_ko] = tokens_list
        
        # 토큰 빈도 계산
        token_freq = Counter(tokens_list)
        disease_token_freq[disnm_ko] = token_freq
        
        # 문서 빈도 업데이트 (각 토큰이 등장하는 질병 수)
        for token in set(tokens_list):  # 중복 제거
            if token in token_doc_count:
                token_doc_count[token] += 1
            else:
                token_doc_count[token] = 1
    
    # TF-IDF 가중치 계산
    total_diseases = len(diseases)
    disease_weights = {}
    
    for disease_id, token_freq in disease_token_freq.items():
        weights = {}
        for token, freq in token_freq.items():
            # TF-IDF 계산
            tf = freq  # 토큰 빈도
            idf = math.log10(total_diseases / token_doc_count[token])  # IDF 계산
            weights[token] = tf * idf
        
        disease_weights[disease_id] = weights
    
    return disease_weights

def calculate_tfidf_similarity(user_tokens, disease_weights):
    """TF-IDF 가중치 기반 유사도 계산"""
    scores = {}
    
    for disease_id, weights in disease_weights.items():
        score = 0
        for token in user_tokens:
            if token in weights:
                score += weights[token]
        
        if score > 0:
            scores[disease_id] = score
    
    return scores

def apply_symptom_bonus(user_tokens, disease_scores, diseases):
    """증상 필드에 포함된 토큰에 가중치 보너스 적용"""
    disease_dict = {disease[0]: disease for disease in diseases}
    
    for disease_id, score in disease_scores.items():
        if disease_id in disease_dict:
            symptoms = disease_dict[disease_id][4]  # 증상 필드
            
            if symptoms:
                symptoms_lower = symptoms.lower()
                bonus = 0
                
                # 사용자 토큰이 증상 필드에 있으면 보너스 부여
                for token in user_tokens:
                    if token and token.lower() in symptoms_lower:
                        bonus += 0.5  # 보너스 가중치 (조정 가능)
                
                if bonus > 0:
                    disease_scores[disease_id] = score + bonus
    
    return disease_scores

# 기존 유사도 함수 (백업)
def calculate_similarity(search_tokens, disease_tokens, symptoms=""):
    search_tokens = [token.lower() for token in search_tokens]
    disease_tokens = [token.lower() for token in disease_tokens]
    
    # 공통 토큰 수 계산
    common_tokens = set(search_tokens) & set(disease_tokens)
    score = len(common_tokens)
    
    # 증상 필드에서 추가 가중치 계산
    if symptoms:
        symptoms_lower = symptoms.lower()
        for token in search_tokens:
            if token in symptoms_lower:
                score += 2
    
    return score