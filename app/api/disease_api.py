# ~/code/backend/app/api/disease_api.py
"""
질병 추천 API - 매칭 로직 개선 버전 (벡터 문제 해결 전까지 사용)
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from pydantic import BaseModel
import json
import re
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer

from app.models.database import get_db

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
    debug_info: Dict[str, Any]

def safe_json_loads(data):
    """PostgreSQL JSONB 데이터를 안전하게 로드"""
    if isinstance(data, str):
        return json.loads(data)
    elif isinstance(data, dict):
        return data
    else:
        return dict(data) if hasattr(data, '__iter__') else data

def extract_morphemes(text: str) -> List[str]:
    """텍스트에서 명사 형태소 추출 (개선된 버전)"""
    try:
        if not text or not text.strip():
            return []
        
        # 특수문자 제거
        cleaned_text = re.sub(r'[^\w\s]', ' ', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # 형태소 분석으로 명사 추출
        nouns = okt.nouns(cleaned_text)
        
        # 의학 관련 키워드 직접 추출 (형태소 분석 보완)
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
        
        logger.debug(f"형태소 분석: '{text}' -> 명사={nouns}, 의학키워드={medical_keywords}, 최종={filtered_tokens}")
        return filtered_tokens
        
    except Exception as e:
        logger.error(f"형태소 분석 중 오류: {str(e)}")
        return []

def load_medical_mappings(db: Session) -> Dict[str, str]:
    """의학 용어 매핑 사전 로드"""
    global _medical_mappings
    
    try:
        if _medical_mappings:  # 이미 로드됨
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

def find_matching_expressions(user_tokens: List[str], 
                            mapping_dict: Dict[str, str],
                            min_overlap: int = 1) -> List[str]:
    """
    사용자 토큰과 일상표현 간 겹치는 토큰이 임계값 이상인 경우 의학용어 반환 (개선된 버전)
    """
    try:
        matched_medical_terms = []
        
        logger.debug(f"매칭 시도 - 사용자 토큰: {user_tokens}")
        
        for common_expr, medical_term in mapping_dict.items():
            # 일상표현도 형태소 분석
            common_tokens = extract_morphemes(common_expr)
            
            # 정확한 매칭 우선 확인
            exact_match = False
            for user_token in user_tokens:
                for common_token in common_tokens:
                    # 정확한 매칭 (길이 2 이상인 경우만)
                    if len(user_token) >= 2 and len(common_token) >= 2 and user_token == common_token:
                        exact_match = True
                        break
                if exact_match:
                    break
            
            if exact_match:
                matched_medical_terms.append(medical_term)
                logger.debug(f"정확 매칭: '{common_expr}' -> '{medical_term}'")
                continue
            
            # 부분 매칭 (더 엄격한 조건)
            meaningful_match = False
            for user_token in user_tokens:
                for common_token in common_tokens:
                    # 부분 매칭 조건을 더 엄격하게
                    if (len(user_token) >= 3 and len(common_token) >= 3 and 
                        (user_token in common_token or common_token in user_token)):
                        # 단순히 "통"만 겹치는 것은 제외
                        if user_token not in ['통', '아프', '아픔'] and common_token not in ['통', '아프', '아픔']:
                            meaningful_match = True
                            break
                if meaningful_match:
                    break
            
            # 겹치는 토큰 수 계산 (길이 2 이상인 토큰만)
            valid_user_tokens = [t for t in user_tokens if len(t) >= 2]
            valid_common_tokens = [t for t in common_tokens if len(t) >= 2]
            overlap_tokens = set(valid_user_tokens) & set(valid_common_tokens)
            overlap_count = len(overlap_tokens)
            
            # 최종 매칭 조건: 정확한 매칭 또는 의미있는 부분매칭 또는 2개 이상 토큰 겹침
            if meaningful_match or overlap_count >= 2:
                matched_medical_terms.append(medical_term)
                logger.debug(f"부분 매칭: '{common_expr}' -> '{medical_term}' (겹침: {overlap_count}, 의미매칭: {meaningful_match}, 겹친토큰: {overlap_tokens})")
        
        # 중복 제거
        unique_terms = list(set(matched_medical_terms))
        logger.debug(f"최종 매칭된 의학용어: {unique_terms}")
        
        return unique_terms
        
    except Exception as e:
        logger.error(f"표현 매칭 중 오류: {str(e)}")
        return []

def load_tfidf_components(db: Session) -> bool:
    """TF-IDF 벡터화에 필요한 컴포넌트들 로드 (수정된 버전)"""
    global _tfidf_vectorizer, _tfidf_vocabulary, _tfidf_idf_weights
    
    try:
        if _tfidf_vectorizer is not None:  # 이미 로드됨
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
        
        logger.info(f"메타데이터 타입 확인: vocabulary={type(result.vocabulary)}, idf_weights={type(result.idf_weights)}")
        
        # JSONB 데이터 안전하게 로드
        _tfidf_vocabulary = safe_json_loads(result.vocabulary)
        _tfidf_idf_weights = safe_json_loads(result.idf_weights)
        
        logger.info(f"파싱 완료: vocabulary_size={len(_tfidf_vocabulary)}, idf_weights_size={len(_tfidf_idf_weights)}")
        
        # 어휘사전 샘플 확인 (디버깅용)
        sample_vocab = dict(list(_tfidf_vocabulary.items())[:10])
        logger.info(f"어휘사전 샘플: {sample_vocab}")
        
        # 한글 토큰 개수 확인
        korean_tokens = [word for word in _tfidf_vocabulary.keys() 
                        if any('\uAC00' <= c <= '\uD7A3' for c in word)]
        logger.info(f"한글 토큰 개수: {len(korean_tokens)}")
        logger.info(f"한글 토큰 샘플: {korean_tokens[:10]}")
        
        # TfidfVectorizer 재구성
        _tfidf_vectorizer = TfidfVectorizer(
            min_df=result.min_df or 2,
            max_df=result.max_df or 0.8,
            max_features=result.max_features,  # None일 수 있음
            ngram_range=(1, 1),  # 1-gram만 사용
            token_pattern=None,
            tokenizer=lambda text: text.split(),
            lowercase=False,
            norm='l2',
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
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
        import traceback
        logger.error(f"상세 에러: {traceback.format_exc()}")
        return False

def create_fallback_vector(tokens: List[str]) -> np.ndarray:
    """vocabulary 매칭이 실패할 경우 사용하는 fallback 벡터"""
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
            '피로': 0.6, '무기력': 0.6
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
                logger.debug(f"Fallback 벡터: '{token}' -> index {idx}, weight {weight}")
        
        # 정규화
        if total_weight > 0:
            vector = vector / np.linalg.norm(vector)
            logger.debug(f"Fallback 벡터 생성: non_zero={np.count_nonzero(vector)}")
        
        return vector
        
    except Exception as e:
        logger.error(f"Fallback 벡터 생성 실패: {str(e)}")
        return np.zeros(len(_tfidf_vocabulary) if _tfidf_vocabulary else 1000)

def vectorize_tokens(tokens: List[str]) -> Optional[np.ndarray]:
    """토큰 리스트를 TF-IDF 벡터로 변환 (개선된 버전)"""
    try:
        if not tokens or not _tfidf_vocabulary:
            logger.warning("토큰이 없거나 어휘사전이 로드되지 않음")
            return None
        
        logger.debug(f"벡터화 시도 - 입력 토큰: {tokens}")
        
        # 직접 어휘사전에서 토큰 매칭 (더 엄격한 매칭)
        valid_tokens = []
        matched_info = []
        
        for token in tokens:
            # 1. 정확한 매칭 (우선순위)
            if token in _tfidf_vocabulary:
                valid_tokens.append(token)
                matched_info.append(f"정확매칭: {token} -> {_tfidf_vocabulary[token]}")
            # 2. 길이 3 이상인 토큰만 부분 매칭 시도
            elif len(token) >= 3:
                found_match = False
                best_match = None
                max_overlap = 0
                
                for vocab_word in _tfidf_vocabulary.keys():
                    if len(vocab_word) >= 3:
                        # 더 엄격한 부분 매칭 조건
                        if token in vocab_word:
                            overlap = len(token)
                            if overlap > max_overlap:
                                max_overlap = overlap
                                best_match = vocab_word
                                found_match = True
                        elif vocab_word in token and len(vocab_word) >= 3:
                            overlap = len(vocab_word)
                            if overlap > max_overlap:
                                max_overlap = overlap
                                best_match = vocab_word
                                found_match = True
                
                if found_match and best_match:
                    valid_tokens.append(best_match)
                    matched_info.append(f"부분매칭: {token} -> {best_match} (겹침길이: {max_overlap})")
                else:
                    matched_info.append(f"매칭실패: {token}")
            else:
                matched_info.append(f"길이부족: {token} (길이: {len(token)})")
        
        logger.debug(f"매칭 결과: {matched_info}")
        
        if not valid_tokens:
            logger.warning(f"유효한 토큰 없음. 원본 토큰: {tokens}")
            return create_fallback_vector(tokens)
        
        logger.debug(f"유효한 토큰들: {valid_tokens}")
        
        # 수동 TF-IDF 벡터 생성
        vector_size = len(_tfidf_vocabulary)
        vector = np.zeros(vector_size)
        
        # 토큰 빈도 계산
        token_counts = {}
        for token in valid_tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        total_tokens = len(valid_tokens)
        
        # TF-IDF 계산
        for token, count in token_counts.items():
            if token in _tfidf_vocabulary:
                idx = _tfidf_vocabulary[token]
                tf = count / total_tokens
                idf = _tfidf_idf_weights.get(token, 1.0)
                
                # Sublinear TF 적용 (scikit-learn과 동일)
                if tf > 0:
                    tf = 1 + np.log(tf)
                
                vector[idx] = tf * idf
                logger.debug(f"벡터 설정: {token} (idx={idx}) tf={tf:.3f} idf={idf:.3f} value={tf*idf:.3f}")
        
        # L2 정규화
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            logger.debug(f"벡터 생성 완료: non_zero={np.count_nonzero(vector)}, norm={norm:.4f}")
            return vector
        else:
            logger.warning("벡터 norm이 0입니다. fallback 벡터 사용")
            return create_fallback_vector(tokens)
        
    except Exception as e:
        logger.error(f"벡터화 중 오류: {str(e)}")
        import traceback
        logger.error(f"상세 에러: {traceback.format_exc()}")
        return create_fallback_vector(tokens)

def load_disease_vectors(db: Session, limit: int = 100) -> List[Dict[str, Any]]:
    """질병 벡터들을 DB에서 로드 (수정된 버전)"""
    global _disease_vectors_cache
    
    try:
        if _disease_vectors_cache:  # 캐시된 데이터 있음
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
        
        results = db.execute(query, {"limit": limit * 2}).fetchall()  # 여유분 로드
        
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
                
            except Exception as vector_error:
                logger.warning(f"벡터 파싱 실패 (ID: {row.disease_id}): {str(vector_error)}")
                continue
        
        logger.info(f"질병 벡터 로드 완료: {len(_disease_vectors_cache)}개")
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
    """질병 추천 메인 엔드포인트"""
    try:
        logger.info(f"질병 추천 요청: {request.original_text}")
        
        # 1. 초기화
        if not initialize_all_components(db):
            raise HTTPException(status_code=500, detail="시스템 초기화 실패")
        
        debug_info = {
            "step1_morphemes": {},
            "step2_mapping": {},
            "step3_vectorization": {},
            "step4_similarity": {}
        }
        
        # 2. 긍정/부정 세그먼트 형태소 분석
        positive_morphemes = []
        for segment in request.positive:
            morphemes = extract_morphemes(segment)
            positive_morphemes.extend(morphemes)
        
        negative_morphemes = []
        for segment in request.negative:
            morphemes = extract_morphemes(segment)
            negative_morphemes.extend(morphemes)
        
        debug_info["step1_morphemes"] = {
            "positive_segments": request.positive,
            "negative_segments": request.negative,
            "positive_morphemes": positive_morphemes,
            "negative_morphemes": negative_morphemes
        }
        
        # 3. 일상표현 매칭
        matched_medical_terms_pos = find_matching_expressions(
            positive_morphemes, _medical_mappings, min_overlap=1
        )
        matched_medical_terms_neg = find_matching_expressions(
            negative_morphemes, _medical_mappings, min_overlap=1
        )
        
        # 4. 최종 토큰 조합 (의학용어 + 매칭되지 않은 형태소)
        final_positive_tokens = list(set(matched_medical_terms_pos + positive_morphemes))
        final_negative_tokens = list(set(matched_medical_terms_neg + negative_morphemes))
        
        debug_info["step2_mapping"] = {
            "matched_medical_pos": matched_medical_terms_pos,
            "matched_medical_neg": matched_medical_terms_neg,
            "final_positive_tokens": final_positive_tokens,
            "final_negative_tokens": final_negative_tokens
        }
        
        # 5. 실시간 벡터화
        positive_vector = vectorize_tokens(final_positive_tokens)
        negative_vector = vectorize_tokens(final_negative_tokens)
        
        debug_info["step3_vectorization"] = {
            "positive_vector_nonzero": int(np.count_nonzero(positive_vector)) if positive_vector is not None else 0,
            "negative_vector_nonzero": int(np.count_nonzero(negative_vector)) if negative_vector is not None else 0,
            "vocabulary_size": len(_tfidf_vocabulary)
        }
        
        # 6. 질병 벡터와 유사도 계산
        disease_vectors = load_disease_vectors(db, limit=100)

        recommendations = []
        for i, disease in enumerate(disease_vectors):
            # 긍정 유사도
            pos_similarity = 0.0
            if positive_vector is not None:
                pos_similarity = abs(float(np.dot(positive_vector, disease['vector'])))  # abs() 추가
                
                # 디버깅: 편두통 질병만 상세 로그
                if '편두통' in disease['disease_name_ko']:
                    logger.info(f"🔍 편두통 디버깅:")
                    logger.info(f"  질병명: {disease['disease_name_ko']}")
                    logger.info(f"  사용자 벡터 shape: {positive_vector.shape}")
                    logger.info(f"  질병 벡터 shape: {disease['vector'].shape}")
                    logger.info(f"  사용자 벡터 nonzero: {np.count_nonzero(positive_vector)}")
                    logger.info(f"  질병 벡터 nonzero: {np.count_nonzero(disease['vector'])}")
                    logger.info(f"  내적 결과: {pos_similarity}")
                    logger.info(f"  사용자 벡터 norm: {np.linalg.norm(positive_vector)}")
                    logger.info(f"  질병 벡터 norm: {np.linalg.norm(disease['vector'])}")
            
            # 부정 유사도
            neg_similarity = 0.0
            if negative_vector is not None:
                neg_similarity = float(np.dot(negative_vector, disease['vector']))
            
            # 최종 점수 (긍정 - 부정*0.3) - 부정 가중치 낮춤
            final_score = pos_similarity - (neg_similarity * 0.3)
            
            # 최소 점수 임계값 낮춤 (0.001 이상)
            if final_score > 0.001:
                recommendations.append(DiseaseRecommendation(
                    disease_id=disease['disease_id'],
                    disease_name_ko=disease['disease_name_ko'],
                    department=disease.get('department'),
                    similarity_score=pos_similarity,
                    final_score=final_score,
                    matched_tokens=final_positive_tokens
                ))
        
        # 7. 정렬 및 상위 10개만 반환
        recommendations.sort(key=lambda x: x.final_score, reverse=True)
        recommendations = recommendations[:10]
        
        debug_info["step4_similarity"] = {
            "total_candidates": len(disease_vectors),
            "above_zero": len(recommendations),
            "top_3_scores": [r.final_score for r in recommendations[:3]],
            "positive_vector_norm": float(np.linalg.norm(positive_vector)) if positive_vector is not None else 0,
            "sample_disease_vector_norms": [float(np.linalg.norm(d['vector'])) for d in disease_vectors[:3]]
        }
        
        logger.info(f"추천 완료: {len(recommendations)}개 결과")
        
        return DiseaseRecommendResponse(
            recommendations=recommendations,
            debug_info=debug_info
        )
        
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
        success = initialize_all_components(db)
        
        return {
            "initialization_success": success,
            "is_initialized": _is_initialized,
            "medical_mappings_count": len(_medical_mappings),
            "vocabulary_size": len(_tfidf_vocabulary),
            "disease_vectors_count": len(_disease_vectors_cache),
            "tfidf_ready": _tfidf_vectorizer is not None
        }
        
    except Exception as e:
        logger.error(f"수동 초기화 중 오류: {str(e)}", exc_info=True)
        return {"error": str(e)}

@router.get("/test-db")
async def test_database_connection(db: Session = Depends(get_db)):
    """데이터베이스 연결 및 테이블 확인"""
    try:
        # 매핑 테이블 확인
        mapping_query = text("SELECT COUNT(*) FROM medical_term_mappings")
        mapping_count = db.execute(mapping_query).scalar()
        
        # TF-IDF 메타데이터 확인
        metadata_query = text("SELECT COUNT(*) FROM tfidf_metadata")
        metadata_count = db.execute(metadata_query).scalar()
        
        # 질병 벡터 확인
        vector_query = text("SELECT COUNT(*) FROM disease_vectors")
        vector_count = db.execute(vector_query).scalar()
        
        # 샘플 메타데이터 조회
        sample_metadata_query = text("""
            SELECT 
                id,
                feature_count,
                min_df,
                max_df,
                max_features,
                pg_typeof(vocabulary) as vocab_type,
                pg_typeof(idf_weights) as idf_type
            FROM tfidf_metadata 
            ORDER BY created_at DESC 
            LIMIT 1
        """)
        sample_metadata = db.execute(sample_metadata_query).fetchone()
        
        # 어휘사전에서 직접 한글 토큰 확인
        vocab_check_query = text("""
            SELECT 
                COUNT(*) as total_keys
            FROM (
                SELECT jsonb_object_keys(vocabulary) as word
                FROM tfidf_metadata 
                ORDER BY created_at DESC 
                LIMIT 1
            ) t
        """)
        vocab_count = db.execute(vocab_check_query).scalar()
        
        return {
            "database_connection": "OK",
            "table_counts": {
                "medical_mappings": int(mapping_count),
                "tfidf_metadata": int(metadata_count),
                "disease_vectors": int(vector_count),
                "vocabulary_keys": int(vocab_count) if vocab_count else 0
            },
            "sample_metadata": {
                "id": sample_metadata.id if sample_metadata else None,
                "feature_count": sample_metadata.feature_count if sample_metadata else None,
                "vocabulary_type": sample_metadata.vocab_type if sample_metadata else None,
                "idf_weights_type": sample_metadata.idf_type if sample_metadata else None
            } if sample_metadata else None
        }
        
    except Exception as e:
        logger.error(f"DB 테스트 중 오류: {str(e)}")
        return {"error": str(e)}

@router.get("/vocabulary-debug")
async def debug_vocabulary(db: Session = Depends(get_db)):
    """어휘사전 디버깅 엔드포인트"""
    try:
        # 초기화 확인
        if not _is_initialized:
            initialize_all_components(db)
        
        # 어휘사전 분석
        total_size = len(_tfidf_vocabulary)
        korean_words = []
        english_words = []
        medical_terms = []
        
        for word, idx in list(_tfidf_vocabulary.items())[:100]:  # 처음 100개만
            if any('\uAC00' <= c <= '\uD7A3' for c in word):
                korean_words.append((word, idx))
            elif word.isalpha() and all(ord(c) < 256 for c in word):
                english_words.append((word, idx))
            
            # 의학용어 확인
            if any(keyword in word for keyword in ['통', '열', '기침', '복통', '구토', '설사']):
                medical_terms.append((word, idx))
        
        # 특정 의학용어 검색
        target_words = ['두통', '복통', '설사', '기침', '발열', '열', '구토']
        found_targets = {}
        for target in target_words:
            found_targets[target] = []
            for word, idx in _tfidf_vocabulary.items():
                if target in word or word == target:
                    found_targets[target].append((word, idx))
        
        return {
            "vocabulary_stats": {
                "total_size": total_size,
                "korean_sample": korean_words[:20],
                "english_sample": english_words[:20],
                "medical_terms": medical_terms[:20]
            },
            "target_search": found_targets,
            "idf_weights_sample": {word: _tfidf_idf_weights.get(word, 0) for word, _ in medical_terms[:10]}
        }
        
    except Exception as e:
        logger.error(f"어휘사전 디버깅 중 오류: {str(e)}")
        return {"error": str(e)}

@router.post("/test-vectorization")
async def test_vectorization(
    tokens: List[str],
    db: Session = Depends(get_db)
):
    """토큰 벡터화 테스트 엔드포인트"""
    try:
        # 초기화 확인
        if not _is_initialized:
            initialize_all_components(db)
        
        # 벡터화 테스트
        vector = vectorize_tokens(tokens)
        
        # 결과 분석
        result = {
            "input_tokens": tokens,
            "vector_created": vector is not None,
            "vector_norm": float(np.linalg.norm(vector)) if vector is not None else 0,
            "non_zero_count": int(np.count_nonzero(vector)) if vector is not None else 0,
            "vocabulary_size": len(_tfidf_vocabulary)
        }
        
        # 토큰별 매칭 정보
        token_matching = {}
        for token in tokens:
            if token in _tfidf_vocabulary:
                token_matching[token] = {
                    "status": "exact_match",
                    "index": _tfidf_vocabulary[token],
                    "idf_weight": _tfidf_idf_weights.get(token, 0)
                }
            else:
                # 부분 매칭 확인
                partial_matches = []
                for vocab_word in list(_tfidf_vocabulary.keys())[:1000]:  # 처음 1000개만 확인
                    if token in vocab_word or vocab_word in token:
                        partial_matches.append(vocab_word)
                        if len(partial_matches) >= 5:  # 최대 5개까지만
                            break
                
                token_matching[token] = {
                    "status": "partial_match" if partial_matches else "no_match",
                    "partial_matches": partial_matches
                }
        
        result["token_matching"] = token_matching
        
        return result
        
    except Exception as e:
        logger.error(f"벡터화 테스트 중 오류: {str(e)}")
        return {"error": str(e)}

@router.get("/version-check")
async def check_version():
    """현재 코드 버전 확인"""
    return {
        "version": "best_working_v1.0",
        "timestamp": "2025-05-22-21:25",
        "features": [
            "improved_matching_logic",
            "strict_partial_matching", 
            "medical_term_prioritization",
            "fallback_vector_support",
            "comprehensive_debugging"
        ]
    }