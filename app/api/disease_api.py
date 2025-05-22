# app/api/disease_api.py
# 디렉토리: backend/app/api/

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Set
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime

from app.models.database import get_db, DatabaseService
from sqlalchemy.orm import Session

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# 요청 모델 (입력 API 출력 형태에 맞춤)
class DiseaseRecommendRequest(BaseModel):
    original_text: str = Field(..., description="원본 증상 텍스트")
    positive: List[str] = Field(..., description="긍정 세그먼트 리스트")
    negative: List[str] = Field(default=[], description="부정 세그먼트 리스트")

# 응답 모델
class DiseaseInfo(BaseModel):
    disease_id: str
    disease_name: str
    similarity_score: float
    department: str
    symptoms: str
    therapy: str
    definition: str

class DiseaseRecommendResponse(BaseModel):
    diseases: List[DiseaseInfo]
    departments: List[str]  # 병원 API용
    disease_names: List[str]  # 의약품 API용
    processed_at: str

# NLP 서비스 클래스 (토큰 기반 유사 매칭 적용)
class NLPService:
    def __init__(self, db: Session):
        self.db = db
        self._mapping_cache = None
    
    def get_medical_term_mappings(self) -> Dict[str, str]:
        """일상용어-의학용어 매핑을 DB에서 가져오거나 캐시에서 반환"""
        if self._mapping_cache is None:
            self._mapping_cache = DatabaseService.get_medical_mappings(self.db)
        return self._mapping_cache
    
    def simple_tokenize(self, text: str) -> List[str]:
        """간단한 토큰화 (공백 및 구두점 기준)"""
        if not text or not text.strip():
            return []
        
        # 기본적인 정제 및 토큰화
        text = text.strip()
        text = text.replace(',', ' ').replace('.', ' ').replace('?', ' ').replace('!', ' ')
        text = text.replace('은', ' ').replace('는', ' ').replace('이', ' ').replace('가', ' ')
        text = text.replace('을', ' ').replace('를', ' ').replace('에', ' ').replace('와', ' ')
        text = text.replace('과', ' ').replace('도', ' ').replace('만', ' ')
        
        tokens = [token.strip() for token in text.split() if token.strip() and len(token.strip()) > 1]
        
        logger.debug(f"토큰화: '{text}' → {tokens}")
        return tokens
    
    def fuzzy_medical_mapping(self, input_tokens: List[str]) -> List[str]:
        """토큰 기반 유사 매칭으로 의학용어 변환"""
        if not input_tokens:
            return []
        
        mappings = self.get_medical_term_mappings()
        mapped_results = []
        used_tokens = set()
        
        logger.info(f"입력 토큰: {input_tokens}")
        
        # 각 매핑 항목과 비교
        for common_term, medical_term in mappings.items():
            # 매핑 테이블의 일상표현도 토큰화
            mapping_tokens = set(self.simple_tokenize(common_term))
            input_token_set = set(input_tokens)
            
            # 교집합 계산
            intersection = mapping_tokens & input_token_set
            
            # 임계값 설정 (1개 이상 매칭 또는 매핑 토큰의 50% 이상)
            min_threshold = 1
            ratio_threshold = len(mapping_tokens) * 0.5
            threshold = max(min_threshold, ratio_threshold)
            
            if len(intersection) >= threshold and intersection:
                mapped_results.append(medical_term)
                used_tokens.update(intersection)
                logger.info(f"매핑 성공: '{common_term}' → '{medical_term}' (교집합: {intersection})")
        
        # 매칭되지 않은 토큰들도 추가 (의학 용어일 수 있음)
        for token in input_tokens:
            if token not in used_tokens and len(token) > 1:
                mapped_results.append(token)
        
        # 중복 제거
        final_results = list(dict.fromkeys(mapped_results))  # 순서 유지하면서 중복 제거
        
        logger.info(f"매핑 결과: {input_tokens} → {final_results}")
        return final_results
    
    def process_segments(self, positive_segments: List[str], negative_segments: List[str]) -> Dict[str, List[str]]:
        """긍정/부정 세그먼트를 처리하여 의학용어로 변환"""
        
        # 긍정 세그먼트 처리
        positive_tokens = []
        for segment in positive_segments:
            tokens = self.simple_tokenize(segment)
            positive_tokens.extend(tokens)
        
        # 부정 세그먼트 처리
        negative_tokens = []
        for segment in negative_segments:
            tokens = self.simple_tokenize(segment)
            negative_tokens.extend(tokens)
        
        # 토큰 기반 의학용어 매핑 적용
        mapped_positive = self.fuzzy_medical_mapping(positive_tokens)
        mapped_negative = self.fuzzy_medical_mapping(negative_tokens)
        
        logger.info(f"처리 완료 - 긍정: {len(mapped_positive)}개, 부정: {len(mapped_negative)}개 토큰")
        
        return {
            'positive_tokens': mapped_positive,
            'negative_tokens': mapped_negative
        }

# TF-IDF 서비스 클래스
class TFIDFService:
    def __init__(self, db: Session):
        self.db = db
        self._metadata = None
        self._vocabulary = None
        self._idf_weights = None
    
    def load_tfidf_metadata(self):
        """TF-IDF 메타데이터 로드"""
        if self._metadata is None:
            metadata = DatabaseService.get_tfidf_metadata(self.db)
            
            # JSONB 데이터 처리
            if isinstance(metadata['vocabulary'], dict):
                self._vocabulary = metadata['vocabulary']
            else:
                self._vocabulary = json.loads(metadata['vocabulary'])
            
            if isinstance(metadata['idf_weights'], dict):
                self._idf_weights = metadata['idf_weights']
            else:
                self._idf_weights = json.loads(metadata['idf_weights'])
            
            logger.info(f"TF-IDF 메타데이터 로드 완료: 어휘 {len(self._vocabulary)}개")
    
    def vectorize_tokens(self, tokens: List[str]) -> np.ndarray:
        """토큰 리스트를 TF-IDF 벡터로 변환 (개선된 버전)"""
        self.load_tfidf_metadata()
        
        if not tokens:
            logger.warning("빈 토큰 리스트 - 0벡터 반환")
            return np.zeros(len(self._vocabulary))
        
        # 토큰 빈도 계산
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # TF-IDF 벡터 생성
        vector = np.zeros(len(self._vocabulary))
        total_tokens = len(tokens)
        matched_tokens = 0
        
        for token, count in token_counts.items():
            if token in self._vocabulary:
                vocab_idx = self._vocabulary[token]
                
                # TF 계산 (sublinear_tf=True 적용)
                tf = 1.0 + np.log(count)
                
                # IDF 가중치 적용
                idf = float(self._idf_weights.get(str(vocab_idx), 0))
                
                # TF-IDF 값
                if idf > 0:  # IDF가 0이 아닌 경우만
                    vector[vocab_idx] = tf * idf
                    matched_tokens += 1
                
                logger.debug(f"토큰 매칭: '{token}' → idx:{vocab_idx}, tf:{tf:.3f}, idf:{idf:.3f}")
        
        # L2 정규화 (0벡터 방지)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            logger.info(f"벡터화 완료 - 매칭된 토큰: {matched_tokens}/{len(token_counts)}, 노름: {norm:.4f}")
        else:
            logger.warning(f"0벡터 생성 - 어휘사전에 매칭되지 않음: {list(token_counts.keys())}")
        
        return vector
    
    def get_disease_vectors(self) -> List[Dict]:
        """모든 질병 벡터 로드"""
        diseases = DatabaseService.get_disease_vectors(self.db)
        self.load_tfidf_metadata()
        
        processed_diseases = []
        for disease in diseases:
            # 벡터 데이터 처리
            vector_data = disease['tfidf_vector']
            if isinstance(vector_data, dict):
                sparse_vector = vector_data
            else:
                sparse_vector = json.loads(vector_data)
            
            # 희소 벡터를 전체 벡터로 복원
            full_vector = np.zeros(len(self._vocabulary))
            for idx, value in sparse_vector.items():
                if int(idx) < len(full_vector):  # 인덱스 범위 확인
                    full_vector[int(idx)] = float(value)
            
            processed_diseases.append({
                'disease_id': disease['disease_id'],
                'disease_name': disease['disease_name'],
                'department': disease['department'],
                'definition': disease['definition'],
                'symptoms': disease['symptoms'],
                'therapy': disease['therapy'],
                'vector': full_vector,
                'original_norm': disease['vector_norm']
            })
        
        logger.info(f"질병 벡터 로드 완료: {len(processed_diseases)}개")
        return processed_diseases
    
    def calculate_similarities(self, user_vector: np.ndarray, negative_vector: np.ndarray, 
                             disease_vectors: List[Dict], negative_weight: float = 0.3) -> List[Dict]:
        """유사도 계산 (개선된 버전)"""
        similarities = []
        
        # 벡터 노름 확인
        user_norm = np.linalg.norm(user_vector)
        neg_norm = np.linalg.norm(negative_vector)
        
        logger.info(f"유사도 계산 시작 - 사용자 벡터 노름: {user_norm:.4f}, 부정 벡터 노름: {neg_norm:.4f}")
        
        if user_norm == 0:
            logger.warning("사용자 벡터가 0벡터 - 모든 유사도가 0이 됩니다")
        
        for disease in disease_vectors:
            disease_vector = disease['vector']
            disease_norm = np.linalg.norm(disease_vector)
            
            if disease_norm == 0:
                # 질병 벡터가 0인 경우 스킵
                continue
            
            # 긍정 유사도 계산
            if user_norm > 0 and disease_norm > 0:
                positive_sim = np.dot(user_vector, disease_vector) / (user_norm * disease_norm)
            else:
                positive_sim = 0.0
            
            # 부정 유사도 계산
            if neg_norm > 0 and disease_norm > 0:
                negative_sim = np.dot(negative_vector, disease_vector) / (neg_norm * disease_norm)
            else:
                negative_sim = 0.0
            
            # 최종 유사도 = 긍정 유사도 - (부정 유사도 × 가중치)
            final_similarity = positive_sim - (negative_sim * negative_weight)
            final_similarity = max(0, final_similarity)  # 음수 방지
            
            similarities.append({
                'disease_id': disease['disease_id'],
                'disease_name': disease['disease_name'],
                'department': disease['department'],
                'definition': disease['definition'],
                'symptoms': disease['symptoms'],
                'therapy': disease['therapy'],
                'similarity_score': float(final_similarity),
                'positive_sim': float(positive_sim),
                'negative_sim': float(negative_sim)
            })
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # 상위 5개 로깅
        top_5 = similarities[:5]
        logger.info(f"상위 5개 질병 유사도:")
        for i, disease in enumerate(top_5):
            logger.info(f"{i+1}. {disease['disease_name']}: {disease['similarity_score']:.4f} "
                       f"(긍정:{disease['positive_sim']:.4f}, 부정:{disease['negative_sim']:.4f})")
        
        return similarities

# 의존성 주입 함수들
def get_nlp_service(db: Session = Depends(get_db)) -> NLPService:
    return NLPService(db)

def get_tfidf_service(db: Session = Depends(get_db)) -> TFIDFService:
    return TFIDFService(db)

# API 엔드포인트
@router.post("/api/disease", response_model=DiseaseRecommendResponse)
async def recommend_diseases(
    request: DiseaseRecommendRequest,
    nlp_service: NLPService = Depends(get_nlp_service),
    tfidf_service: TFIDFService = Depends(get_tfidf_service)
):
    """
    증상 기반 질병 추천 API (토큰 기반 유사 매칭 적용)
    
    입력 예시:
    {
        "original_text": "배는 괜찮은데 목이 아프고 속이 매스꺼워요 기침이 나아지지 않아요",
        "positive": ["목이 아프고", "속이 매스꺼워요", "기침이 나아지지 않아요"],
        "negative": ["배는 괜찮은데"]
    }
    """
    
    try:
        logger.info(f"=== 질병 추천 요청 시작 ===")
        logger.info(f"원본: {request.original_text}")
        logger.info(f"긍정 세그먼트: {request.positive}")
        logger.info(f"부정 세그먼트: {request.negative}")
        
        # 1. 세그먼트 처리 및 의학용어 매핑 (개선된 토큰 기반 매칭)
        processed = nlp_service.process_segments(request.positive, request.negative)
        
        logger.info(f"매핑 후 긍정 토큰: {processed['positive_tokens']}")
        logger.info(f"매핑 후 부정 토큰: {processed['negative_tokens']}")
        
        # 2. TF-IDF 벡터화
        positive_vector = tfidf_service.vectorize_tokens(processed['positive_tokens'])
        negative_vector = tfidf_service.vectorize_tokens(processed['negative_tokens'])
        
        # 3. 질병 벡터 로드
        disease_vectors = tfidf_service.get_disease_vectors()
        
        # 4. 유사도 계산
        similarities = tfidf_service.calculate_similarities(
            positive_vector, negative_vector, disease_vectors
        )
        
        # 5. 상위 5개 질병 선택
        top_diseases = similarities[:5]
        
        # 6. 응답 데이터 구성
        disease_infos = []
        departments = set()
        disease_names = []
        
        for disease in top_diseases:
            disease_info = DiseaseInfo(
                disease_id=disease['disease_id'],
                disease_name=disease['disease_name'],
                similarity_score=round(disease['similarity_score'] * 100, 2),  # 퍼센트로 변환
                department=disease['department'],
                symptoms=disease['symptoms'],
                therapy=disease['therapy'],
                definition=disease['definition']
            )
            disease_infos.append(disease_info)
            
            # 진료과 수집 (병원 API용)
            if disease['department']:
                dept_list = [dept.strip() for dept in disease['department'].split(',')]
                departments.update(dept_list)
            
            # 질병명 수집 (의약품 API용)
            disease_names.append(disease['disease_name'])
        
        logger.info(f"=== 질병 추천 완료 ===")
        logger.info(f"추천 질병: {[d.disease_name for d in disease_infos]}")
        
        return DiseaseRecommendResponse(
            diseases=disease_infos,
            departments=list(departments),
            disease_names=disease_names,
            processed_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"질병 추천 실패: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"질병 추천 중 오류가 발생했습니다: {str(e)}")

# 헬스 체크 엔드포인트
@router.get("/api/disease/health")
async def health_check():
    """질병 추천 API 헬스 체크"""
    return {
        "status": "healthy",
        "service": "disease_recommendation",
        "timestamp": datetime.now().isoformat()
    }

# 테스트용 엔드포인트 (개선된 디버깅 정보 포함)
@router.post("/api/disease/test")
async def test_disease_recommendation(
    request: DiseaseRecommendRequest,
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """테스트용 질병 추천 엔드포인트 (상세 정보 포함)"""
    
    try:
        # 처리된 토큰 확인
        processed = nlp_service.process_segments(request.positive, request.negative)
        
        # 매핑 테이블 정보
        mappings = nlp_service.get_medical_term_mappings()
        
        return {
            "original_text": request.original_text,
            "positive_segments": request.positive,
            "negative_segments": request.negative,
            "processed_positive_tokens": processed['positive_tokens'],
            "processed_negative_tokens": processed['negative_tokens'],
            "available_mappings_count": len(mappings),
            "sample_mappings": dict(list(mappings.items())[:5]),  # 샘플 5개
            "debug_info": {
                "positive_token_count": len(processed['positive_tokens']),
                "negative_token_count": len(processed['negative_tokens']),
                "mapping_applied": True
            }
        }
        
    except Exception as e:
        logger.error(f"테스트 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))