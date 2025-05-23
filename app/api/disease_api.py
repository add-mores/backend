# ~/backend/app/api/disease_api.py
"""
질병 추천 API - 정확도 및 성능 개선 버전

주요 개선사항:
1. 형태소 분석 정확도 향상 (의학 용어 패턴 확장)
2. 의학 용어 매핑 알고리즘 효율화
3. 불용어 처리 강화
4. 유사도 계산 최적화
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any, Optional, Tuple
import logging
import numpy as np
from pydantic import BaseModel
import json
import re
import math
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import lru_cache
import time
from app.models.database import get_db

# 로거 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API 라우터 초기화
router = APIRouter(prefix="/api", tags=["disease"])

# 전역 변수 (메모리 캐시용)
_medical_mappings = {}
_tfidf_vectorizer = None
_tfidf_vocabulary = {}
_tfidf_idf_weights = {}
_disease_vectors_cache = []
_is_initialized = False

# Okt 형태소 분석기 초기화
okt = Okt()

# 요청/응답 모델
class DiseaseRecommendRequest(BaseModel):
    original_text: str
    positive: List[str] = []
    negative: List[str] = []

class DiseaseRecommendation(BaseModel):
    disease_id: str
    disease_name_ko: str
    department: Optional[str] = None
    similarity_score: float
    final_score: float
    matched_tokens: List[str] = []

class DiseaseRecommendResponse(BaseModel):
    recommendations: List[DiseaseRecommendation]
    debug_info: Optional[Dict[str, Any]] = None

# 중요 짧은 의학 용어 예외 목록
IMPORTANT_SHORT_TERMS = {
    "눈", "귀", "코", "입", "목", "손", "발", "팔", "다리", "피부", "잇몸", "턱", "몸",
    "털", "살", "힘", "열", "통", "염", "혈", "압", "암", "뇌", "폐", "간", "심", "장",
    "위", "담", "장", "신", "비", "콩", "땀", "골", "근", "탈", "종", "균", "약"
}

# 의학 전문 불용어
MEDICAL_STOPWORDS = [
    '증상', '병원', '질환', '진단', '검사', '치료', '약', '의사', '처방',
    '몸', '상태', '느낌', '것', '정도', '때', '그', '저', '나', '나의', '내',
    '가다', '오다', '하다', '되다', '있다', '없다', '같다', '좀', '많이',
    '매우', '너무', '아주', '계속', '지금', '요즘', '가끔', '항상', '자주',
    '것', '수', '등', '중', '등등', '및', '또', '또는', '혹은', '때문'
]

# 의학 용어 동의어 사전
MEDICAL_SYNONYMS = {
    '구역': '구토',
    '구역질': '구토',
    '메스꺼움': '구토',
    '토함': '구토',
    '머리아픔': '두통',
    '머리통증': '두통',
    '열감': '발열',
    '체온상승': '발열',
    '숨참': '호흡곤란',
    '복부불편감': '복통',
    '위통': '복통',
    '목아픔': '인후통',
    '목따가움': '인후통',
    '어지러움': '현기증',
    '기력저하': '피로',
    '무기력증': '피로'
}

# 의학 관련 명사 사전
MEDICAL_NOUNS = {
    '가래', '기침', '열', '통증', '두통', '편두통', '발열', '콧물', '재채기',
    '오한', '인후통', '인후염', '비염', '구토', '설사', '복통', '소화불량',
    '현기증', '어지러움', '피로', '근육통', '관절통', '발진', '홍반', '부종',
    '불면', '호흡', '숨', '가슴', '흉통', '목', '코', '배', '위', '관절',
    '근육', '피부', '두드러기', '식욕', '체온', '메스꺼움', '구역', '어깨',
    '요통', '허리', '허리통증', '이명', '귀', '눈', '시력', '충혈', '붓기',
    '안구', '시력', '결막', '각막', '망막', '시신경', '시야', '초점', '충혈', 
    '안구건조', '안구피로', '시력저하', '눈물', '눈곱', '안경', '렌즈'
}

class MedicalAwareVectorizer:
    """의학 지식을 반영한 벡터화 구현"""
    
    def __init__(self, vocabulary, idf_weights):
        """
        Parameters:
            vocabulary: 어휘사전 {토큰: 인덱스}
            idf_weights: IDF 가중치 {토큰: 가중치}
        """
        self.vocabulary = vocabulary
        self.idf_weights = idf_weights
        self.vector_size = len(vocabulary)
        
        # 의학적 관계 그래프 (기본값)
        self.medical_relationships = self._default_medical_graph()
        
        # 카테고리별 대표 질병
        self.category_representatives = {
            "감기_독감": ["감기", "독감", "인플루엔자", "상기도감염"],
            "호흡기": ["폐렴", "기관지염", "천식", "기관지확장증"],
            "소화기": ["위염", "장염", "위장염", "대장염", "소화불량"],
            "신경계": ["편두통", "뇌염", "수막염", "뇌졸중"],
            "심혈관": ["고혈압", "협심증", "심근경색", "부정맥"],
            "피부": ["두드러기", "피부염", "습진", "건선"]
        }
    
    def _default_medical_graph(self):
        """기본 의학 관계 그래프"""
        return {
            # 감기 관련 증상-질병 관계
            "기침": {
                "감기": 0.9, "독감": 0.8, "기관지염": 0.7, "폐렴": 0.6, 
                "천식": 0.5, "후두염": 0.5
            },
            "콧물": {
                "감기": 0.9, "알레르기비염": 0.8, "부비동염": 0.7, 
                "독감": 0.7, "상기도감염": 0.8
            },
            "인후통": {
                "감기": 0.9, "인두염": 0.8, "편도염": 0.8, "후두염": 0.7, 
                "독감": 0.7, "상기도감염": 0.8
            },
            "발열": {
                "감기": 0.7, "독감": 0.9, "폐렴": 0.8, "수막염": 0.7, 
                "말라리아": 0.8, "패혈증": 0.8
            },
            # 소화기 관련 증상-질병 관계
            "복통": {
                "위염": 0.8, "장염": 0.8, "위궤양": 0.7, "충수염": 0.9, 
                "담낭염": 0.7, "췌장염": 0.7
            },
            "구토": {
                "위장염": 0.9, "식중독": 0.8, "편두통": 0.6, "뇌진탕": 0.6, 
                "임신": 0.6, "멀미": 0.7
            },
            "설사": {
                "장염": 0.9, "위장염": 0.8, "식중독": 0.8, "과민성대장증후군": 0.7, 
                "크론병": 0.6, "감염성설사": 0.9
            },
            # 두통 관련 증상-질병 관계
            "두통": {
                "편두통": 0.9, "긴장성두통": 0.8, "군발두통": 0.8, "부비동염": 0.7, 
                "수막염": 0.7, "뇌종양": 0.6
            },
            # 눈 관련 증상-질병 관계
            "눈": {
                "결막염": 0.8, "안구건조증": 0.8, "알레르기성결막염": 0.7,
                "녹내장": 0.6, "백내장": 0.5
            },
            "안구피로": {
                "안구건조증": 0.9, "근시": 0.7, "난시": 0.7, "노안": 0.6,
                "안구피로증후군": 0.9
            }
        }
    
    def vectorize(self, tokens):
        """토큰 리스트를 의학적 맥락을 고려한 벡터로 변환"""
        if not tokens:
            return None
        
        # 1. 기본 TF-IDF 벡터 생성
        vector = np.zeros(self.vector_size)
        
        # 2. 토큰 카운팅 (TF 계산용)
        token_counts = {}
        for token in tokens:
            if token in self.vocabulary:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # 3. 벡터 구성
        for token, count in token_counts.items():
            if token in self.vocabulary:
                idx = self.vocabulary[token]
                tf = count / len(tokens) if len(tokens) > 0 else 0
                idf = self.idf_weights.get(token, 1.0)
                vector[idx] = tf * idf
        
        # 4. 정규화
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def get_category_similarity(self, tokens):
        """토큰에 가장 유사한 의학 카테고리 반환"""
        if not tokens:
            return None, 0.0
            
        # 각 카테고리별 대표 질병과의 유사도 계산
        category_scores = {}
        
        for category, diseases in self.category_representatives.items():
            # 증상 토큰과 카테고리 대표 질병들 간의 관계 강도 합산
            score = 0.0
            matched_count = 0
            
            for token in tokens:
                if token in self.medical_relationships:
                    for disease in diseases:
                        if disease in self.medical_relationships[token]:
                            score += self.medical_relationships[token][disease]
                            matched_count += 1
            
            # 정규화 (매칭된 증상 수로 나눔)
            if matched_count > 0:
                category_scores[category] = score / matched_count
            else:
                category_scores[category] = 0.0
        
        # 가장 높은 점수의 카테고리 반환
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            return best_category[0], best_category[1]
        
        return None, 0.0
    
def safe_json_loads(data):
    """PostgreSQL JSONB 데이터를 안전하게 로드"""
    if isinstance(data, str):
        return json.loads(data)
    elif isinstance(data, dict):
        return data
    else:
        return dict(data) if hasattr(data, '__iter__') else data

def preprocess_text(text: str) -> str:
    """텍스트 전처리 (특수문자 제거, 정규화)"""
    if not text or not isinstance(text, str):
        return ""
        
    # 특수문자 제거 및 공백 정규화
    cleaned_text = re.sub(r'[^\w\s]', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip().lower()
    
    return cleaned_text

def trace_tokens(stage: str, tokens: List[str]):
    """토큰 처리 과정 로그"""
    if tokens:
        tokens_str = ", ".join([f"'{t}'" for t in tokens[:10]])  # 처음 10개만
        if len(tokens) > 10:
            tokens_str += f" ... (총 {len(tokens)}개)"
        logger.info(f"[{stage}] 토큰: [{tokens_str}]")
    else:
        logger.warning(f"[{stage}] 토큰 없음!")

def extract_morphemes(text: str) -> List[str]:
    """형태소 분석 (불용어 처리 강화)"""
    try:
        if not text or not text.strip():
            logger.warning(f"입력 텍스트 없음: '{text}'")
            return []
        
        logger.info(f"형태소 분석 입력: '{text}'")
        
        # 텍스트 전처리
        cleaned_text = preprocess_text(text)
        
        # 형태소 분석으로 명사 추출
        nouns = okt.nouns(cleaned_text)
        trace_tokens("명사 추출", nouns)
        
        # 의학 관련 키워드 직접 추출
        medical_keywords = []
        # 패턴 매칭으로 의학 용어 추출
        for term in MEDICAL_NOUNS:
            if term in cleaned_text:
                medical_keywords.append(term)
        
        trace_tokens("의학 용어", medical_keywords)
        
        # 명사 + 의학 키워드 결합
        all_tokens = list(set(nouns + medical_keywords))
        trace_tokens("전체 토큰", all_tokens)
        
        # 불용어 필터링
        filtered_tokens = []
        removed_tokens = []
        
        for token in all_tokens:
            # 불용어 체크
            if token.lower() in [w.lower() for w in MEDICAL_STOPWORDS]:
                removed_tokens.append(f"{token}(불용어)")
                continue
                
            # 길이 체크 (중요 용어는 예외 처리)
            if len(token) < 2:
                if token in IMPORTANT_SHORT_TERMS:
                    # 중요 짧은 용어는 유지
                    filtered_tokens.append(token)
                    continue
                removed_tokens.append(f"{token}(짧음)")
                continue
                
            # 숫자만 있는 토큰 제외
            if token.isdigit():
                removed_tokens.append(f"{token}(숫자)")
                continue
                
            # 유효한 토큰 추가
            filtered_tokens.append(token)
        
        trace_tokens("필터링 후", filtered_tokens)
        
        # 동의어 처리
        normalized_tokens = []
        for token in filtered_tokens:
            normalized_tokens.append(MEDICAL_SYNONYMS.get(token, token))
        
        # 중복 제거
        final_tokens = list(set(normalized_tokens))
        trace_tokens("최종 토큰", final_tokens)
        
        return final_tokens
        
    except Exception as e:
        logger.error(f"형태소 분석 중 오류: {str(e)}")
        return []

def extract_core_combinations(text: str) -> List[Tuple[str, str]]:
    """텍스트에서 명사+서술어 핵심 조합 추출"""
    combinations = []
    
    # 조사 제거 전처리
    cleaned_text = re.sub(r'(이|가|은|는|을|를|에|의|로|게|도|만|까지|부터|와|과|며|고|자|서)(\s|$)', ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # 명사 추출
    nouns = okt.nouns(cleaned_text)
    
    # 서술어 패턴 (동사, 형용사 어간)
    verb_patterns = [
        r'아프', r'쑤시', r'아픈', r'쑤신', r'통증', r'통증이', 
        r'아프다', r'쑤시다', r'아픔', r'통증은', r'통증을',
        r'건조', r'가렵', r'따갑', r'간지', r'붓', r'부어',
        r'피로', r'피곤', r'지치', r'무기력', r'힘들'
    ]
    
    # 각 명사와 서술어 패턴 조합 확인
    for noun in nouns:
        if len(noun) < 1:  # 짧은 명사도 포함
            continue
            
        for pattern in verb_patterns:
            if re.search(pattern, text):
                combinations.append((noun, pattern))
    
    return combinations

def improved_medical_mapping(original_text: str, mapping_dict: Dict[str, str]) -> List[str]:
    """구문 중심의 의학 용어 매핑 함수 (핵심 조합 인식 추가)"""
    mapped_terms = []
    normalized_text = preprocess_text(original_text).lower()
    
    # 1단계: 전체 구문 매핑 (가장 높은 우선순위)
    for common_expr, medical_term in mapping_dict.items():
        if common_expr.lower() in normalized_text:
            if medical_term not in mapped_terms:
                mapped_terms.append(medical_term)
                logger.debug(f"전체 구문 매핑: '{common_expr}' → '{medical_term}'")
    
    # 2단계: 핵심 조합 매핑 (명사+서술어)
    text_combinations = extract_core_combinations(original_text)
    for common_expr, medical_term in mapping_dict.items():
        expr_combinations = extract_core_combinations(common_expr)
        
        # 원문과 표현 간 핵심 조합 일치 여부 확인
        for text_combo in text_combinations:
            for expr_combo in expr_combinations:
                # 명사와 서술어 모두 일치하는지
                if text_combo[0] == expr_combo[0] and expr_combo[1] in text_combo[1]:
                    if medical_term not in mapped_terms:
                        mapped_terms.append(medical_term)
                        logger.debug(f"핵심 조합 매핑: '{text_combo}' → '{medical_term}'")
                        break
            else:
                continue
            break
    
    # 3단계: 특별 케이스 매핑
    # 특정 신체 부위 + 증상 조합
    body_symptom_map = {
        ('머리', '아프'): '두통',
        ('머리', '지끈'): '편두통',
        ('머리', '통증'): '두통',
        ('눈', '피로'): '안구 피로',
        ('눈', '건조'): '안구건조증',
        ('눈', '가려움'): '결막염',
        ('눈', '충혈'): '결막염',
        ('목', '아프'): '인후통',
        ('목', '따가움'): '인후통',
        ('코', '막힘'): '비염',
        ('코', '콧물'): '비염',
        ('배', '아프'): '복통',
        ('배', '통증'): '복통',
        ('관절', '아프'): '관절통',
        ('관절', '통증'): '관절통',
        ('근육', '아프'): '근육통',
        ('근육', '통증'): '근육통'
    }
    
    # 텍스트에서 명사 추출
    nouns = okt.nouns(original_text)
    
    # 각 명사와 서술어 패턴 조합으로 매핑
    for noun in nouns:
        for symptom_pattern in ['아프', '통증', '지끈', '피로', '건조', '가려움', '충혈', '따가움', '막힘']:
            if symptom_pattern in original_text.lower():
                key = (noun, symptom_pattern)
                if key in body_symptom_map:
                    mapped_term = body_symptom_map[key]
                    if mapped_term not in mapped_terms:
                        mapped_terms.append(mapped_term)
                        logger.debug(f"신체-증상 매핑: '{key}' → '{mapped_term}'")
    
    # 4단계: 기존 핵심 구문 매핑 (조사, 어미 제거)
    for common_expr, medical_term in mapping_dict.items():
        # 일상 표현에서 핵심 부분만 추출 (조사, 어미 제거)
        # 조사 제거 패턴 확장
        core_expr = re.sub(r'(이|가|은|는|을|를|에|의|로|게|도|요|다)(\s|$)', ' ', common_expr).strip().lower()
        if len(core_expr) >= 4 and core_expr in normalized_text and medical_term not in mapped_terms:
            mapped_terms.append(medical_term)
            logger.debug(f"핵심 구문 매핑: '{core_expr}' → '{medical_term}'")
    
    # 5단계: 의미 단위 매핑 (이전과 동일)
    tokens = extract_morphemes(original_text)
    
    # 중요 의미 조합 사전
    meaning_combinations = {
        ("눈", "피로"): "안구 피로",
        ("눈", "건조"): "안구건조증",
        ("눈", "충혈"): "결막염",
        ("눈", "가려움"): "알레르기성 결막염",
        ("머리", "아프다"): "두통",
        ("머리", "지끈"): "편두통",
        ("코", "막힘"): "비염",
        ("코", "콧물"): "비염",
        ("목", "아프다"): "인후통",
        ("기침", "가래"): "기관지염",
        ("배", "아프다"): "복통",
        ("구토", "설사"): "장염"
    }
    
    # 토큰 조합으로 매핑
    token_pairs = [(tokens[i], tokens[j]) for i in range(len(tokens)) for j in range(i+1, len(tokens))]
    for token_pair in token_pairs:
        # 순서 무관 매칭을 위해 두 가지 순서 모두 확인
        for combination in [token_pair, (token_pair[1], token_pair[0])]:
            if combination in meaning_combinations:
                term = meaning_combinations[combination]
                if term not in mapped_terms:
                    mapped_terms.append(term)
                    logger.debug(f"의미 조합 매핑: '{combination}' → '{term}'")
    
    return mapped_terms

def load_medical_mappings(db: Session) -> Dict[str, str]:
    """의학 용어 매핑 사전 로드"""
    global _medical_mappings
    
    try:
        if _medical_mappings:
            return _medical_mappings
        
        logger.info("의학 용어 매핑 사전 로드 시작")
        
        query = text("""
            SELECT common_term, medical_term 
            FROM medical_term_mappings 
            ORDER BY common_term
        """)
        
        results = db.execute(query).fetchall()
        
        for row in results:
            _medical_mappings[row.common_term.strip()] = row.medical_term.strip()
        
        logger.info(f"매핑 사전 로드 완료: {len(_medical_mappings)}개")
        return _medical_mappings
        
    except Exception as e:
        logger.error(f"매핑 사전 로드 실패: {str(e)}")
        return {}

def load_tfidf_components(db: Session) -> bool:
    """TF-IDF 벡터화에 필요한 컴포넌트들 로드"""
    global _tfidf_vectorizer, _tfidf_vocabulary, _tfidf_idf_weights
    
    try:
        if _tfidf_vectorizer is not None:
            return True
        
        logger.info("TF-IDF 컴포넌트 로드 시작")
        
        # 메타데이터 조회
        metadata_query = text("""
            SELECT 
                vocabulary,
                idf_weights,
                feature_count,
                min_df,
                max_df,
                max_features
            FROM tfidf_metadata 
            ORDER BY created_at DESC 
            LIMIT 1
        """)
        
        result = db.execute(metadata_query).fetchone()
        if not result:
            logger.error("TF-IDF 메타데이터 없음")
            return False
        
        # JSONB 데이터 안전하게 로드
        _tfidf_vocabulary = safe_json_loads(result.vocabulary)
        _tfidf_idf_weights = safe_json_loads(result.idf_weights)
        
        logger.info(f"파싱 완료: vocabulary_size={len(_tfidf_vocabulary)}, idf_weights_size={len(_tfidf_idf_weights)}")
        
        # TfidfVectorizer 재구성
        _tfidf_vectorizer = TfidfVectorizer(
            min_df=result.min_df or 2,
            max_df=result.max_df or 0.8,
            max_features=result.max_features,
            ngram_range=(1, 1),
            token_pattern=None,
            tokenizer=lambda text: text.split(),
            lowercase=False,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False  # 음수 방지를 위해 False로 설정
        )
        
        # 어휘사전과 IDF 가중치 설정
        _tfidf_vectorizer.vocabulary_ = _tfidf_vocabulary
        _tfidf_vectorizer.idf_ = np.array([
            _tfidf_idf_weights.get(word, 1.0) 
            for word in _tfidf_vocabulary.keys()
        ])
        
        logger.info(f"TF-IDF 컴포넌트 로드 완료: vocabulary_size={len(_tfidf_vocabulary)}")
        return True
        
    except Exception as e:
        logger.error(f"TF-IDF 컴포넌트 로드 실패: {str(e)}")
        return False

def create_fallback_vector(tokens: List[str]) -> np.ndarray:
    """fallback 벡터 생성"""
    try:
        vector_size = len(_tfidf_vocabulary) if _tfidf_vocabulary else 1000
        vector = np.zeros(vector_size)
        
        # 의학용어별 고정 가중치
        medical_weights = {
            '두통': 0.8, '편두통': 0.8, '머리': 0.6,
            '발열': 0.8, '열': 0.7, '고열': 0.8,
            '기침': 0.7, '가래': 0.6,
            '복통': 0.7, '배': 0.5,
            '인후통': 0.7, '목': 0.5,
            '현기증': 0.7, '어지러': 0.6,
            '구토': 0.7, '메스꺼': 0.6,
            '설사': 0.7, '변비': 0.6,
            '피로': 0.6, '무기력': 0.6,
            '눈': 0.7, '안구피로': 0.8
        }
        
        # 토큰별 가중치 적용
        total_weight = 0
        for token in tokens:
            if token in medical_weights:
                weight = medical_weights[token]
                # 해시 기반으로 벡터 위치 결정
                idx = abs(hash(token)) % vector_size
                vector[idx] = weight
                total_weight += weight
        
        # 정규화
        if total_weight > 0:
            vector = vector / np.linalg.norm(vector)
        
        return vector
        
    except Exception as e:
        logger.error(f"Fallback 벡터 생성 실패: {str(e)}")
        return np.zeros(len(_tfidf_vocabulary) if _tfidf_vocabulary else 1000)

def vectorize_tokens(tokens: List[str]) -> Optional[np.ndarray]:
    """토큰 리스트를 TF-IDF 벡터로 변환"""
    try:
        if not tokens:
            return None
        if not _tfidf_vectorizer:
            return None
        
        # 벡터화할 토큰 필터링
        filtered_tokens = [t for t in tokens if t in _tfidf_vocabulary or t in IMPORTANT_SHORT_TERMS]
        
        # 벡터화할 토큰이 없으면 fallback 벡터 사용
        if not filtered_tokens:
            return create_fallback_vector(tokens)
        
        # 토큰 문서로 변환
        document = ' '.join(filtered_tokens)
        
        # 벡터화
        vector = _tfidf_vectorizer.transform([document]).toarray().flatten()
        
        # 정규화
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            return vector
        else:
            return create_fallback_vector(tokens)
        
    except Exception as e:
        logger.error(f"벡터화 중 오류: {str(e)}")
        return create_fallback_vector(tokens)

def load_disease_vectors(db: Session, limit: int = 100) -> List[Dict[str, Any]]:
    """질병 벡터들을 DB에서 로드"""
    global _disease_vectors_cache
    
    try:
        if _disease_vectors_cache:
            return _disease_vectors_cache[:limit]
        
        logger.info("질병 벡터 로드 시작")
        
        query = text("""
            SELECT 
                disease_id,
                disease_name_ko,
                department,
                tfidf_vector,
                vector_norm
            FROM disease_vectors 
            WHERE vector_norm > 0 
            ORDER BY non_zero_count DESC
            LIMIT :limit
        """)
        
        results = db.execute(query, {"limit": limit * 2}).fetchall()
        
        for row in results:
            try:
                # JSONB 형태의 sparse vector를 dense vector로 변환
                sparse_vector = safe_json_loads(row.tfidf_vector)
                dense_vector = np.zeros(len(_tfidf_vocabulary))
                
                for idx_str, value in sparse_vector.items():
                    idx = int(idx_str)
                    if 0 <= idx < len(_tfidf_vocabulary):
                        dense_vector[idx] = float(value)
                
                # 정규화 확인
                norm = np.linalg.norm(dense_vector)
                if norm > 0:
                    dense_vector = dense_vector / norm
                
                    _disease_vectors_cache.append({
                        'disease_id': row.disease_id,
                        'disease_name_ko': row.disease_name_ko,
                        'department': row.department,
                        'vector': dense_vector
                    })
                
            except Exception as e:
                logger.warning(f"벡터 파싱 실패 (ID: {row.disease_id}): {str(e)}")
                continue
        
        logger.info(f"질병 벡터 로드 완료: {len(_disease_vectors_cache)}개")
        return _disease_vectors_cache[:limit]
        
    except Exception as e:
        logger.error(f"질병 벡터 로드 실패: {str(e)}")
        return []

def calculate_enhanced_similarity(
    positive_tokens: List[str],
    disease_name: str,
    cosine_similarity: float
) -> float:
    """핵심 증상 기반 향상된 유사도 계산"""
    
    # 코사인 유사도가 0 이하면 즉시 0 반환
    if cosine_similarity <= 0:
        return 0.0
        
    # 질병 카테고리별 핵심 증상
    CORE_SYMPTOMS = {
        "감기_독감": ["기침", "콧물", "인후통", "발열", "두통", "근육통", "오한", "재채기", "미열", "목감기", "코막힘"],
        "호흡기": ["기침", "가래", "호흡곤란", "천명음", "콧물", "인후통", "코막힘", "목감기", "코피", "흉통", "기도", "호흡", "숨참"],
        "소화기": ["복통", "구토", "설사", "변비", "메스꺼움", "구역", "복부", "소화불량", "속쓰림", "명치통증", "위통", "식욕부진"],
        "두통": ["두통", "편두통", "머리", "어지러움", "현기증", "두통약", "머리아픔", "측두통", "후두통", "전두통"],
        "안과": ["눈", "안구피로", "시력저하", "눈물", "충혈", "결막염", "안구건조", "안구통증", "각막염"]
    }
    
    # 질병명과 카테고리 연결 사전
    COMMON_DISEASES = {
        "감기": "감기_독감", "독감": "감기_독감", "인플루엔자": "감기_독감",
        "비염": "감기_독감", "축농증": "감기_독감", "부비동염": "감기_독감",
        "편도염": "감기_독감", "인두염": "감기_독감", "후두염": "감기_독감",
        "기관지염": "호흡기", "폐렴": "호흡기", "천식": "호흡기",
        "위장염": "소화기", "장염": "소화기", "위염": "소화기",
        "편두통": "두통", "긴장성두통": "두통", "군발두통": "두통",
        "결막염": "안과", "안구건조증": "안과", "각막염": "안과", "녹내장": "안과"
    }
    
    final_score = cosine_similarity
    
    # 질병 이름으로 카테고리 찾기
    category = None
    for disease_key, category_name in COMMON_DISEASES.items():
        if disease_key in disease_name.lower():
            category = category_name
            break
    
    # 카테고리가 있으면 핵심 증상 매칭 확인
    if category:
        core_symptoms = CORE_SYMPTOMS[category]
        matched_core_symptoms = [token for token in positive_tokens if token in core_symptoms]
        match_count = len(matched_core_symptoms)
        
        # 핵심 증상 매칭 비율 계산
        total_core_symptoms = len(core_symptoms)
        match_ratio = match_count / total_core_symptoms if total_core_symptoms > 0 else 0
        
        # 유사도 점수 보정
        symptom_boost = 0
        if match_count > 0:
            # 매칭된 증상 수에 따른 기본 가산점
            symptom_boost = 0.1 * math.log(1 + match_count * 2)
            
            # 매칭 비율에 따른 추가 가산점
            if match_ratio >= 0.3:
                symptom_boost += 0.05
                
            if match_ratio >= 0.5:
                symptom_boost += 0.05
                
            # 카테고리별 특화 보정
            if category == "감기_독감" and match_count >= 2:
                symptom_boost += 0.15
                
            # 감기의 핵심 증상을 모두 가진 경우 추가 부스트
            cold_core_symptoms = ["기침", "콧물", "인후통", "발열"]
            cold_match_count = sum(1 for s in positive_tokens if s in cold_core_symptoms)
            if cold_match_count >= 3:
                symptom_boost += 0.1
            
            # 안과 관련 증상 매칭 (추가)
            if category == "안과" and ("눈" in matched_core_symptoms or "안구피로" in matched_core_symptoms):
                symptom_boost += 0.1
                
            # 최대 가산점 제한
            max_boost = 0.3
            symptom_boost = min(symptom_boost, max_boost)
        
        # 최종 점수 계산
        final_score = cosine_similarity + symptom_boost
    
    return final_score

def initialize_all_components(db: Session) -> bool:
    """모든 컴포넌트 초기화"""
    global _is_initialized
    
    if _is_initialized:
        return True
    
    try:
        logger.info("전체 컴포넌트 초기화 시작")
        
        # 1. 의학 용어 매핑 로드
        mappings = load_medical_mappings(db)
        if not mappings:
            logger.warning("의학 용어 매핑 로드 실패")
        
        # 2. TF-IDF 컴포넌트 로드
        if not load_tfidf_components(db):
            logger.error("TF-IDF 컴포넌트 로드 실패")
            return False
        
        # 3. 질병 벡터 로드
        disease_vectors = load_disease_vectors(db, limit=1000)
        if not disease_vectors:
            logger.error("질병 벡터 로드 실패")
            return False
        
        _is_initialized = True
        logger.info("전체 초기화 완료")
        return True
        
    except Exception as e:
        logger.error(f"초기화 실패: {str(e)}")
        return False
    
@router.post("/disease", response_model=DiseaseRecommendResponse)
async def recommend_diseases(
    request: DiseaseRecommendRequest,
    db: Session = Depends(get_db)
):
    """질병 추천 메인 엔드포인트 (의학적 맥락 고려)"""
    try:
        logger.info(f"질병 추천 요청: '{request.original_text}'")
        start_time = time.time()
        
        # 1. 초기화
        if not initialize_all_components(db):
            raise HTTPException(status_code=500, detail="시스템 초기화 실패")
        
        # 2. 의학 지식 기반 벡터화 초기화
        medical_vectorizer = MedicalAwareVectorizer(
            vocabulary=_tfidf_vocabulary,
            idf_weights=_tfidf_idf_weights
        )
        
        # 3. 긍정/부정 세그먼트 처리
        positive_morphemes = []
        for segment in request.positive:
            morphemes = extract_morphemes(segment)
            positive_morphemes.extend(morphemes)
        
        negative_morphemes = []
        if request.negative:  # 부정 세그먼트가 있을 때만 처리
            for segment in request.negative:
                morphemes = extract_morphemes(segment)
                negative_morphemes.extend(morphemes)
        
        # 4. 일상표현 매핑
        matched_medical_terms_pos = []
        for segment in request.positive:
            mapped_terms = improved_medical_mapping(segment, _medical_mappings)
            matched_medical_terms_pos.extend(mapped_terms)
        
        logger.info(f"[매핑 결과] 원문: '{request.positive[0] if request.positive else ''}' → 매핑: {matched_medical_terms_pos}")
        
        matched_medical_terms_neg = []
        if request.negative:  # 부정 세그먼트가 있을 때만 매핑
            for segment in request.negative:
                mapped_terms = improved_medical_mapping(segment, _medical_mappings)
                matched_medical_terms_neg.extend(mapped_terms)
        
        # 5. 최종 토큰 조합
        final_positive_tokens = list(set(matched_medical_terms_pos + positive_morphemes))
        final_negative_tokens = list(set(matched_medical_terms_neg + negative_morphemes))
        
        # 6. 벡터화
        positive_vector = medical_vectorizer.vectorize(final_positive_tokens)
        
        # 부정 세그먼트가 있을 때만 벡터화
        negative_vector = None
        if final_negative_tokens:
            negative_vector = medical_vectorizer.vectorize(final_negative_tokens)
        
        # 7. 질병 벡터와 유사도 계산
        disease_vectors = load_disease_vectors(db, limit=100)
        logger.info(f"질병 벡터 로드: {len(disease_vectors)}개")

        recommendations = []
        for disease in disease_vectors:
            # 코사인 유사도 계산
            pos_similarity = 0.0
            if positive_vector is not None and disease['vector'] is not None:
                pos_similarity = float(np.dot(positive_vector, disease['vector']))
            
            # 부정 유사도 (부정 벡터가 있을 때만 계산)
            neg_similarity = 0.0
            if negative_vector is not None and disease['vector'] is not None:
                neg_similarity = float(np.dot(negative_vector, disease['vector']))
            
            # 향상된 유사도 계산
            base_score = pos_similarity - (neg_similarity * 0.3)
            final_score = calculate_enhanced_similarity(
                final_positive_tokens, 
                disease['disease_name_ko'],
                base_score
            )
            
            # 최소 점수 임계값 (0.001 이상)
            if final_score > 0.001:
                recommendations.append(DiseaseRecommendation(
                    disease_id=disease['disease_id'],
                    disease_name_ko=disease['disease_name_ko'],
                    department=disease.get('department'),
                    similarity_score=pos_similarity,
                    final_score=final_score,
                    matched_tokens=final_positive_tokens
                ))
        
        # 8. 정렬 및 상위 5개만 반환
        recommendations.sort(key=lambda x: x.final_score, reverse=True)
        top_recommendations = recommendations[:5]
        
        # 9. 처리 시간 및 결과 로깅
        processing_time = time.time() - start_time
        logger.info(f"추천 완료: {len(top_recommendations)}개 결과, 처리 시간: {processing_time:.3f}초")
        
        return DiseaseRecommendResponse(
            recommendations=top_recommendations,
            debug_info={
                "positive_tokens": final_positive_tokens,
                "negative_tokens": final_negative_tokens,
                "processing_time": f"{processing_time:.3f}초"
            }
        )
        
    except Exception as e:
        logger.error(f"질병 추천 중 오류: {str(e)}")
        return {"error": str(e)}

@router.get("/status")
async def get_system_status(db: Session = Depends(get_db)):
    """시스템 상태 확인"""
    try:
        return {
            "is_initialized": _is_initialized,
            "medical_mappings_count": len(_medical_mappings),
            "vocabulary_size": len(_tfidf_vocabulary),
            "disease_vectors_count": len(_disease_vectors_cache),
            "tfidf_ready": _tfidf_vectorizer is not None,
            "version": "2.0.0",
            "last_update": "2025-05-23"
        }
        
    except Exception as e:
        return {"error": str(e)}

@router.post("/initialize")
async def manual_initialize(db: Session = Depends(get_db)):
    """수동 시스템 초기화"""
    try:
        global _is_initialized, _medical_mappings, _tfidf_vocabulary, _tfidf_idf_weights, _disease_vectors_cache
        
        # 기존 캐시 초기화
        _medical_mappings.clear()
        _tfidf_vocabulary.clear()
        _tfidf_idf_weights.clear()
        _disease_vectors_cache.clear()
        _is_initialized = False
        
        # 강제 초기화
        start_time = time.time()
        success = initialize_all_components(db)
        
        return {
            "initialization_success": success,
            "is_initialized": _is_initialized,
            "medical_mappings_count": len(_medical_mappings),
            "vocabulary_size": len(_tfidf_vocabulary),
            "disease_vectors_count": len(_disease_vectors_cache),
            "tfidf_ready": _tfidf_vectorizer is not None,
            "processing_time": f"{time.time() - start_time:.3f}초"
        }
        
    except Exception as e:
        logger.error(f"수동 초기화 중 오류: {str(e)}")
        return {"error": str(e)}