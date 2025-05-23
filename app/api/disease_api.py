# ~/code/backend/app/api/disease_api.py
"""
질병 추천 API - 정리된 버전
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

from app.models.database import get_db

# 로거 설정
logger = logging.getLogger(__name__)

# API 라우터 초기화
router = APIRouter(prefix="/api", tags=["disease"])

# 전역 변수
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

def safe_json_loads(data):
    """PostgreSQL JSONB 데이터를 안전하게 로드"""
    if isinstance(data, str):
        return json.loads(data)
    elif isinstance(data, dict):
        return data
    else:
        return dict(data) if hasattr(data, '__iter__') else data

def extract_morphemes(text: str) -> List[str]:
    """텍스트에서 명사 형태소 추출"""
    try:
        if not text or not text.strip():
            return []
        
        # 특수문자 제거
        cleaned_text = re.sub(r'[^\w\s]', ' ', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # 형태소 분석으로 명사 추출
        nouns = okt.nouns(cleaned_text)
        
        # 의학 관련 키워드 직접 추출
        medical_keywords = []
        medical_patterns = [
            r'머리.*?아프', r'두통', r'머리',
            r'열', r'발열', r'고열',
            r'기침', r'가래', r'콧물',
            r'배.*?아프', r'복통', r'배',
            r'목.*?아프', r'인후통', r'목',
            r'가슴.*?아프', r'흉통', r'가슴',
            r'어지러', r'현기증',
            r'구토', r'토하', r'메스꺼',
            r'설사', r'변비', r'소화불량',
            r'피로', r'무기력', r'권태감'
        ]
        
        for pattern in medical_patterns:
            if re.search(pattern, text):
                if '머리' in pattern or '두통' in pattern:
                    medical_keywords.append('두통')
                elif '열' in pattern or '발열' in pattern:
                    medical_keywords.append('발열')
                elif '기침' in pattern:
                    medical_keywords.append('기침')
                elif '복통' in pattern or '배' in pattern:
                    medical_keywords.append('복통')
                elif '목' in pattern or '인후통' in pattern:
                    medical_keywords.append('인후통')
                elif '가슴' in pattern or '흉통' in pattern:
                    medical_keywords.append('흉통')
                elif '어지러' in pattern or '현기증' in pattern:
                    medical_keywords.append('현기증')
                elif '구토' in pattern or '토하' in pattern or '메스꺼' in pattern:
                    medical_keywords.append('구토')
                elif '설사' in pattern:
                    medical_keywords.append('설사')
                elif '피로' in pattern:
                    medical_keywords.append('피로')
        
        # 명사 + 의학 키워드 결합
        all_tokens = list(set(nouns + medical_keywords))
        
        # 길이 1 이상인 토큰만 필터링
        filtered_tokens = [token for token in all_tokens if len(token) >= 1 and token.strip()]
        
        return filtered_tokens
        
    except Exception as e:
        logger.error(f"형태소 분석 중 오류: {str(e)}")
        return []

def load_medical_mappings(db: Session) -> Dict[str, str]:
    """의학 용어 매핑 사전 로드"""
    global _medical_mappings
    
    try:
        if _medical_mappings:
            return _medical_mappings
        
        query = text("""
            SELECT common_term, medical_term 
            FROM medical_term_mappings 
            ORDER BY common_term
        """)
        
        results = db.execute(query).fetchall()
        
        for row in results:
            _medical_mappings[row.common_term.strip()] = row.medical_term.strip()
        
        return _medical_mappings
        
    except Exception as e:
        logger.error(f"매핑 사전 로드 실패: {str(e)}")
        return {}

def find_phrase_matches(text: str, mapping_dict: Dict[str, str]) -> List[Tuple[str, float]]:
    matches = []
    
    if not text or not text.strip():
        return matches
    
    normalized_text = text.lower()
    
    for common_expr, medical_term in mapping_dict.items():
        # 1. 정확한 매칭
        if common_expr.lower() in normalized_text:
            matches.append((medical_term, 1.0))
            continue
            
        # 2. 어간 기반 매칭 (새로 추가)
        # "목이 아프다" → "목이 아프"로 변환하여 매칭
        common_stem = common_expr.lower().rstrip('다요고며서')
        if len(common_stem) >= 3 and common_stem in normalized_text:
            matches.append((medical_term, 0.95))
            continue
            
        # 3. 핵심 단어 조합 매칭
        # "콧물이 나다" → ["콧물", "나"] 모두 있으면 매칭
        common_words = common_expr.lower().split()
        if len(common_words) >= 2:
            key_words = [w for w in common_words if len(w) >= 2]
            if all(word in normalized_text for word in key_words):
                matches.append((medical_term, 0.9))
    
    return matches

def find_noun_combination_matches(tokens: List[str], mapping_dict: Dict[str, str]) -> List[Tuple[str, float]]:
    """명사 조합 매칭"""
    matches = []
    
    if not tokens or len(tokens) < 2:
        return matches
    
    token_combinations = []
    for i in range(len(tokens)):
        if i < len(tokens) - 1:
            token_combinations.append(' '.join(tokens[i:i+2]))
        if i < len(tokens) - 2:
            token_combinations.append(' '.join(tokens[i:i+3]))
    
    for combo in token_combinations:
        for common_expr, medical_term in mapping_dict.items():
            if combo.lower() in common_expr.lower() or common_expr.lower() in combo.lower():
                score = 0.7 if len(combo) >= 4 else 0.6
                matches.append((medical_term, score))
    
    return matches

def find_individual_matches(tokens: List[str], mapping_dict: Dict[str, str]) -> List[Tuple[str, float]]:
    """개별 명사 매칭"""
    matches = []
    
    if not tokens:
        return matches
    
    for token in tokens:
        if len(token) < 2:
            continue
            
        for common_expr, medical_term in mapping_dict.items():
            if token.lower() == common_expr.lower():
                matches.append((medical_term, 0.5))
                break
            elif len(token) >= 3 and (token.lower() in common_expr.lower() or common_expr.lower() in token.lower()):
                matches.append((medical_term, 0.3))
    
    return matches

def select_representative_tokens(matches: List[Tuple[str, float]]) -> List[str]:
    """카테고리별 대표 토큰 선택"""
    if not matches:
        return []
    
    categories = {
        '두통': [], '발열': [], '인후통': [], '기침': [], '복통': [],
        '어지럼증': [], '구토': [], '설사': [], '피로': [], '기타': []
    }
    
    for term, score in matches:
        categorized = False
        for category in categories.keys():
            if category in term:
                categories[category].append((term, score))
                categorized = True
                break
        
        if not categorized:
            categories['기타'].append((term, score))
    
    final_terms = []
    for category, terms in categories.items():
        if terms:
            terms.sort(key=lambda x: x[1], reverse=True)
            top_score = terms[0][1]
            top_terms = [term for term, score in terms if score == top_score]
            
            if category in ['두통', '발열', '인후통', '기침', '복통']:
                top_terms = top_terms[:1]
            else:
                top_terms = top_terms[:2]
                
            final_terms.extend(top_terms)
    
    return final_terms

def find_matching_expressions(user_tokens: List[str], 
                            user_original_text: str,
                            mapping_dict: Dict[str, str]) -> List[str]:
    """사용자 토큰과 일상표현을 단계적으로 매칭"""
    try:
        if not user_tokens and not user_original_text:
            return []
            
        phrase_matches = find_phrase_matches(user_original_text, mapping_dict)
        
        noun_combination_matches = []
        if len(phrase_matches) < 3:
            noun_combination_matches = find_noun_combination_matches(user_tokens, mapping_dict)
        
        individual_matches = []
        if len(phrase_matches) + len(noun_combination_matches) < 2:
            individual_matches = find_individual_matches(user_tokens, mapping_dict)
        
        all_matches = phrase_matches + noun_combination_matches + individual_matches
        final_matches = select_representative_tokens(all_matches)
        
        return final_matches
        
    except Exception as e:
        logger.error(f"표현 매칭 중 오류: {str(e)}")
        return []

def load_tfidf_components(db: Session) -> bool:
    """TF-IDF 벡터화에 필요한 컴포넌트들 로드"""
    global _tfidf_vectorizer, _tfidf_vocabulary, _tfidf_idf_weights
    
    try:
        if _tfidf_vectorizer is not None:
            return True
        
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
            return False
        
        _tfidf_vocabulary = safe_json_loads(result.vocabulary)
        _tfidf_idf_weights = safe_json_loads(result.idf_weights)
        
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
            sublinear_tf=False
        )
        
        _tfidf_vectorizer.vocabulary_ = _tfidf_vocabulary
        _tfidf_vectorizer.idf_ = np.array([
            _tfidf_idf_weights.get(word, 1.0) 
            for word in _tfidf_vocabulary.keys()
        ])
        
        return True
        
    except Exception as e:
        logger.error(f"TF-IDF 컴포넌트 로드 실패: {str(e)}")
        return False

def create_fallback_vector(tokens: List[str]) -> np.ndarray:
    """fallback 벡터 생성"""
    try:
        vector_size = len(_tfidf_vocabulary) if _tfidf_vocabulary else 1000
        vector = np.zeros(vector_size)
        
        medical_weights = {
            '두통': 0.8, '편두통': 0.8, '머리': 0.6,
            '발열': 0.8, '열': 0.7, '고열': 0.8,
            '기침': 0.7, '가래': 0.6,
            '복통': 0.7, '배': 0.5,
            '인후통': 0.7, '목': 0.5,
            '현기증': 0.7, '어지러': 0.6,
            '구토': 0.7, '메스꺼': 0.6,
            '설사': 0.7, '변비': 0.6,
            '피로': 0.6, '무기력': 0.6
        }
        
        total_weight = 0
        for token in tokens:
            if token in medical_weights:
                weight = medical_weights[token]
                idx = abs(hash(token)) % vector_size
                vector[idx] = weight
                total_weight += weight
        
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
        
        document = ' '.join(tokens)
        vector = _tfidf_vectorizer.transform([document]).toarray().flatten()
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            return vector
        else:
            return create_fallback_vector(tokens)
        
    except Exception as e:
        logger.error(f"벡터화 중 오류: {str(e)}")
        return create_fallback_vector(tokens)

def calculate_enhanced_similarity(
    positive_tokens: List[str],
    disease_name: str,
    cosine_similarity: float
) -> float:
    """핵심 증상 기반 향상된 유사도 계산"""
    
    # 코사인 유사도가 0 이하면 즉시 0 반환
    if cosine_similarity <= 0:
        return 0.0
    CORE_SYMPTOMS = {
        "감기_독감": ["기침", "콧물", "인후통", "발열", "두통", "근육통", "오한", "재채기", "미열", "목", "목감기", "코막힘"],
        "호흡기": ["기침", "가래", "호흡곤란", "천명음", "콧물", "인후통", "코막힘", "목", "목감기", "코", "코피", "흉통", "기도", "호흡", "숨참"],
        "소화기": ["복통", "구토", "설사", "변비", "메스꺼움", "구역", "복부", "소화불량", "속쓰림", "명치통증", "위통", "식욕부진", "토함"],
        "두통": ["두통", "편두통", "머리", "어지러움", "현기증", "두통약", "머리아픔", "측두통", "후두통", "전두통"],
        "발열": ["발열", "열", "고열", "미열", "오한", "체온상승", "열나다", "열감", "체온"],
        "인후통": ["인후통", "목", "목아픔", "목감기", "목칼칼", "목따가움", "인후", "인후염", "편도"],
        "피부": ["발진", "두드러기", "가려움", "홍반", "습진", "피부", "붉음", "물집", "피부염", "피부병"],
        "신경계": ["두통", "어지러움", "현기증", "마비", "경련", "발작", "떨림", "무감각", "저림", "신경통"],
        "관절_근육": ["관절통", "근육통", "요통", "허리통증", "근육", "관절", "허리", "목", "어깨", "무릎"],
        "이비인후과": ["귀통증", "이명", "청력저하", "코막힘", "코피", "인후통", "편도", "부비동", "중이염", "외이도"],
        "안과": ["눈통증", "충혈", "시력저하", "눈부심", "안구건조", "눈가려움", "눈물", "눈곱", "결막염", "각막염"]
    }
    
    COMMON_DISEASES = {
        "감기": "감기_독감", "독감": "감기_독감", "인플루엔자": "감기_독감",
        "비염": "감기_독감", "축농증": "감기_독감", "부비동염": "감기_독감",
        "편도염": "감기_독감", "인두염": "감기_독감", "후두염": "감기_독감",
        "기관지염": "호흡기", "폐렴": "호흡기", "천식": "호흡기",
        "위장염": "소화기", "장염": "소화기", "위염": "소화기",
        "편두통": "두통", "긴장성두통": "두통", "군발두통": "두통",
        "두드러기": "피부", "습진": "피부", "피부염": "피부",
        "관절염": "관절_근육", "근육통": "관절_근육",
        "중이염": "이비인후과", "결막염": "안과"
    }
    
    final_score = cosine_similarity
    
    category = None
    for disease_key, category_name in COMMON_DISEASES.items():
        if disease_key in disease_name:
            category = category_name
            break
    
    if category:
        core_symptoms = CORE_SYMPTOMS[category]
        matched_core_symptoms = [token for token in positive_tokens if token in core_symptoms]
        match_count = len(matched_core_symptoms)
        
        total_core_symptoms = len(core_symptoms)
        match_ratio = match_count / total_core_symptoms if total_core_symptoms > 0 else 0
        
        symptom_boost = 0
        if match_count > 0:
            symptom_boost = 0.1 * math.log(1 + match_count * 2)
            
            if match_ratio >= 0.3:
                symptom_boost += 0.05
                
            if match_ratio >= 0.5:
                symptom_boost += 0.05
                
            if category == "감기_독감" and match_count >= 2:
                symptom_boost += 0.15  # 0.05 → 0.15로 증가
                
            # 감기의 4대 증상을 모두 가진 경우 추가 부스트
            cold_core_symptoms = ["기침", "콧물", "인후통", "발열"]
            cold_match_count = len([s for s in positive_tokens if s in cold_core_symptoms or 
                                   (s == "비루" and "콧물" in cold_core_symptoms) or
                                   (s == "재채기" and "콧물" in cold_core_symptoms)])
            if cold_match_count >= 3:
                symptom_boost += 0.1
                
            if category == "호흡기" and "기침" in matched_core_symptoms:
                symptom_boost += 0.02
                
            if category == "소화기" and "복통" in matched_core_symptoms:
                symptom_boost += 0.02
                
            if category == "두통" and "두통" in matched_core_symptoms:
                symptom_boost += 0.03
                
            max_boost = 0.3
            symptom_boost = min(symptom_boost, max_boost)
        
        final_score = cosine_similarity + symptom_boost
    
    return final_score

def load_disease_vectors(db: Session, limit: int = 100) -> List[Dict[str, Any]]:
    """질병 벡터들을 DB에서 로드"""
    global _disease_vectors_cache
    
    try:
        if _disease_vectors_cache:
            return _disease_vectors_cache[:limit]
        
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
                sparse_vector = safe_json_loads(row.tfidf_vector)
                dense_vector = np.zeros(len(_tfidf_vocabulary))
                
                for idx_str, value in sparse_vector.items():
                    idx = int(idx_str)
                    if 0 <= idx < len(_tfidf_vocabulary):
                        dense_vector[idx] = float(value)
                
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
        
        return _disease_vectors_cache[:limit]
        
    except Exception as e:
        logger.error(f"질병 벡터 로드 실패: {str(e)}")
        return []

def initialize_all_components(db: Session) -> bool:
    """모든 컴포넌트 초기화"""
    global _is_initialized
    
    if _is_initialized:
        return True
    
    try:
        mappings = load_medical_mappings(db)
        if not mappings:
            logger.warning("의학 용어 매핑 로드 실패")
        
        if not load_tfidf_components(db):
            logger.error("TF-IDF 컴포넌트 로드 실패")
            return False
        
        disease_vectors = load_disease_vectors(db, limit=1000)
        if not disease_vectors:
            logger.error("질병 벡터 로드 실패")
            return False
        
        _is_initialized = True
        return True
        
    except Exception as e:
        logger.error(f"초기화 실패: {str(e)}")
        return False

@router.post("/disease", response_model=DiseaseRecommendResponse)
async def recommend_diseases(
    request: DiseaseRecommendRequest,
    db: Session = Depends(get_db)
):
    """질병 추천 메인 엔드포인트"""
    try:
        if not initialize_all_components(db):
            raise HTTPException(status_code=500, detail="시스템 초기화 실패")
        
        # 긍정/부정 세그먼트 형태소 분석
        positive_morphemes = []
        for segment in request.positive:
            morphemes = extract_morphemes(segment)
            positive_morphemes.extend(morphemes)
        
        negative_morphemes = []
        for segment in request.negative:
            morphemes = extract_morphemes(segment)
            negative_morphemes.extend(morphemes)
        
        # 일상표현 매칭
        matched_medical_terms_pos = find_matching_expressions(
            positive_morphemes, 
            request.positive[0] if request.positive else "", 
            _medical_mappings
        )
        
        matched_medical_terms_neg = find_matching_expressions(
            negative_morphemes, 
            request.negative[0] if request.negative else "", 
            _medical_mappings
        )
        
        # 최종 토큰 조합
        final_positive_tokens = list(set(matched_medical_terms_pos + positive_morphemes))
        final_negative_tokens = list(set(matched_medical_terms_neg + negative_morphemes))
        
        # 실시간 벡터화
        positive_vector = vectorize_tokens(final_positive_tokens)
        negative_vector = vectorize_tokens(final_negative_tokens)
        
        # 질병 벡터와 유사도 계산
        disease_vectors = load_disease_vectors(db, limit=100)

        recommendations = []
        for disease in disease_vectors:
            pos_similarity = 0.0
            if positive_vector is not None:
                pos_similarity = float(np.dot(positive_vector, disease['vector']))
            
            neg_similarity = 0.0
            if negative_vector is not None:
                neg_similarity = float(np.dot(negative_vector, disease['vector']))
            
            base_score = pos_similarity - (neg_similarity * 0.3)
            final_score = calculate_enhanced_similarity(
                final_positive_tokens, 
                disease['disease_name_ko'],
                base_score
            )
            
            if final_score > 0.001:
                recommendations.append(DiseaseRecommendation(
                    disease_id=disease['disease_id'],
                    disease_name_ko=disease['disease_name_ko'],
                    department=disease.get('department'),
                    similarity_score=pos_similarity,
                    final_score=final_score,
                    matched_tokens=final_positive_tokens
                ))
        
        recommendations.sort(key=lambda x: x.final_score, reverse=True)
        recommendations = recommendations[:5]  # 상위 5개만 반환
        
        return DiseaseRecommendResponse(recommendations=recommendations)
        
    except Exception as e:
        logger.error(f"질병 추천 중 오류: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_system_status(db: Session = Depends(get_db)):
    """시스템 상태 확인"""
    try:
        return {
            "is_initialized": _is_initialized,
            "medical_mappings_count": len(_medical_mappings),
            "vocabulary_size": len(_tfidf_vocabulary),
            "disease_vectors_count": len(_disease_vectors_cache),
            "tfidf_ready": _tfidf_vectorizer is not None
        }
    except Exception as e:
        return {"error": str(e)}