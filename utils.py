import os
import psycopg2
import json
from dotenv import load_dotenv


load_dotenv()
# DB 연결 함수
def get_db_connection():
    DATABASE_URL = os.getenv("DATABASE_URL")
    conn = psycopg2.connect(DATABASE_URL)
    return conn

# 토큰 처리 함수
def process_tokens(tokens):
    if isinstance(tokens, str):
        try:
            return json.loads(tokens)
        except:
            return tokens.split()
    return tokens

# 유사도 계산 함수
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