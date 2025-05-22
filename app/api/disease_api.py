# app/api/diseases.py
# 디렉토리: backend/app/api/

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging
from datetime import datetime

from app.models.database import get_db
from sqlalchemy.orm import Session
from sqlalchemy import text

# 로깅 설정
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

# NLP 서비스 클래스
class NLPService:
    def __init__(self, db: Session):
        self.db = db
        self._mapping_cache = None
    
    def get_medical_term_mappings(self) -> Dict[str, str]:
        """일상용어-의학용어 매핑을 DB에서 가져오거나 캐시에서 반환"""
        if self._mapping_cache is None:
            mappings = {}
            query = text("SELECT common_term, medical_term FROM medical_term_mappings")
            result = self.db.execute(query)
            
            for row in result:
                mappings[row.common_term] = row.medical_term
            
            self._mapping_cache = mappings
            logger.info(f"의학용어 매핑 로드 완료: {len(mappings)}개")
        
        return self._mapping_cache
    
    def apply_medical_mapping(self, tokens: List[str]) -> List[str]:
        """토큰에 의학용어 매핑 적용"""
        mappings = self.get_medical_term_mappings()
        mapped_tokens = []
        
        for token in tokens:
            if token in mappings:
                mapped_tokens.append(mappings[token])
                logger.debug(f"매핑 적용: {token} → {mappings[token]}")
            else:
                mapped_tokens.append(token)
        
        return mapped_tokens
    
    def simple_tokenize(self, text: str) -> List[str]:
        """간단한 토큰화 (공백 기준)"""
        if not text or not text.strip():
            return []
        
        # 기본적인 정제
        text = text.strip().replace(',', ' ').replace('.', ' ')
        tokens = [token.strip() for token in text.split() if token.strip()]
        
        return tokens
    
    def process_segments(self, positive_segments: List[str], negative_segments: List[str]) -> Dict[str, List[str]]:
        """긍정/부정 세그먼트를 처리하여 의학용어로 변환"""
        
        # 긍정 세그먼트 처리
        positive_tokens = []
        for segment in positive_segments:
            tokens = self.simple_tokenize(segment)
            mapped_tokens = self.apply_medical_mapping(tokens)
            positive_tokens.extend(mapped_tokens)
        
        # 부정 세그먼트 처리
        negative_tokens = []
        for segment in negative_segments:
            tokens = self.simple_tokenize(segment)
            mapped_tokens = self.apply_medical_mapping(tokens)
            negative_tokens.extend(mapped_tokens)
        
        logger.info(f"처리 완료 - 긍정: {len(positive_tokens)}개, 부정: {len(negative_tokens)}개 토큰")
        
        return {
            'positive_tokens': positive_tokens,
            'negative_tokens': negative_tokens
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
            query = text("SELECT vocabulary, idf_weights, feature_count FROM tfidf_metadata ORDER BY id DESC LIMIT 1")
            result = self.db.execute(query).fetchone()
            
            if not result:
                raise HTTPException(status_code=500, detail="TF-IDF 메타데이터를 찾을 수 없습니다.")
            
            # JSONB 데이터 처리
            if isinstance(result.vocabulary, dict):
                self._vocabulary = result.vocabulary
            else:
                self._vocabulary = json.loads(result.vocabulary)
            
            if isinstance(result.idf_weights, dict):
                self._idf_weights = result.idf_weights
            else:
                self._idf_weights = json.loads(result.idf_weights)
            
            logger.info(f"TF-IDF 메타데이터 로드 완료: 어휘 {len(self._vocabulary)}개")
    
    def vectorize_tokens(self, tokens: List[str]) -> np.ndarray:
        """토큰 리스트를 TF-IDF 벡터로 변환"""
        self.load_tfidf_metadata()
        
        if not tokens:
            return np.zeros(len(self._vocabulary))
        
        # 토큰 빈도 계산
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # TF-IDF 벡터 생성
        vector = np.zeros(len(self._vocabulary))
        total_tokens = len(tokens)
        
        for token, count in token_counts.items():
            if token in self._vocabulary:
                vocab_idx = self._vocabulary[token]
                
                # TF 계산 (sublinear_tf=True 적용)
                tf = 1.0 + np.log(count)
                
                # IDF 가중치 적용
                idf = float(self._idf_weights.get(str(vocab_idx), 0))
                
                # TF-IDF 값
                vector[vocab_idx] = tf * idf
        
        # L2 정규화
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def get_disease_vectors(self) -> List[Dict]:
        """모든 질병 벡터 로드"""
        query = text("""
            SELECT dv.disease_id, dv.disease_name_ko, dv.department, 
                   dv.tfidf_vector, dv.vector_norm,
                   d.def as definition, d.symptoms, d.therapy
            FROM disease_vectors dv
            LEFT JOIN disv2 d ON CAST(SUBSTRING(dv.disease_id FROM 9) AS INTEGER) = d.id
            WHERE dv.vector_norm > 0
            ORDER BY dv.disease_id
        """)
        
        result = self.db.execute(query)
        diseases = []
        
        for row in result:
            # 벡터 데이터 처리
            vector_data = row.tfidf_vector
            if isinstance(vector_data, dict):
                sparse_vector = vector_data
            else:
                sparse_vector = json.loads(vector_data)
            
            # 희소 벡터를 전체 벡터로 복원
            full_vector = np.zeros(len(self._vocabulary))
            for idx, value in sparse_vector.items():
                full_vector[int(idx)] = float(value)
            
            diseases.append({
                'disease_id': row.disease_id,
                'disease_name': row.disease_name_ko,
                'department': row.department or '',
                'definition': row.definition or '',
                'symptoms': row.symptoms or '',
                'therapy': row.therapy or '',
                'vector': full_vector
            })
        
        logger.info(f"질병 벡터 로드 완료: {len(diseases)}개")
        return diseases
    
    def calculate_similarities(self, user_vector: np.ndarray, negative_vector: np.ndarray, 
                             disease_vectors: List[Dict], negative_weight: float = 0.3) -> List[Dict]:
        """유사도 계산"""
        similarities = []
        
        for disease in disease_vectors:
            disease_vector = disease['vector']
            
            # 긍정 유사도 계산
            positive_sim = cosine_similarity([user_vector], [disease_vector])[0][0]
            
            # 부정 유사도 계산
            negative_sim = cosine_similarity([negative_vector], [disease_vector])[0][0]
            
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
                'similarity_score': float(final_similarity)
            })
        
        # 유사도 순으로 정렬
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        
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
    증상 기반 질병 추천 API
    
    입력 예시:
    {
        "original_text": "배는 괜찮은데 목이 아프고 속이 매스꺼워요 기침이 나아지지 않아요",
        "positive": ["목이 아프고", "속이 매스꺼워요", "기침이 나아지지 않아요"],
        "negative": ["배는 괜찮은데"]
    }
    """
    
    try:
        logger.info(f"질병 추천 요청 - 원본: {request.original_text}")
        logger.info(f"긍정 세그먼트: {request.positive}")
        logger.info(f"부정 세그먼트: {request.negative}")
        
        # 1. 세그먼트 처리 및 의학용어 매핑
        processed = nlp_service.process_segments(request.positive, request.negative)
        
        # 2. TF-IDF 벡터화
        positive_vector = tfidf_service.vectorize_tokens(processed['positive_tokens'])
        negative_vector = tfidf_service.vectorize_tokens(processed['negative_tokens'])
        
        logger.info(f"벡터화 완료 - 긍정 벡터 노름: {np.linalg.norm(positive_vector):.4f}")
        
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
        
        logger.info(f"추천 완료 - 상위 5개 질병: {[d.disease_name for d in disease_infos]}")
        
        return DiseaseRecommendResponse(
            diseases=disease_infos,
            departments=list(departments),
            disease_names=disease_names,
            processed_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"질병 추천 실패: {str(e)}")
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

# 테스트용 엔드포인트
@router.post("/api/disease/test")
async def test_disease_recommendation(
    request: DiseaseRecommendRequest,
    nlp_service: NLPService = Depends(get_nlp_service)
):
    """테스트용 질병 추천 엔드포인트 (상세 정보 포함)"""
    
    # 처리된 토큰 확인
    processed = nlp_service.process_segments(request.positive, request.negative)
    
    return {
        "original_text": request.original_text,
        "positive_segments": request.positive,
        "negative_segments": request.negative,
        "processed_positive_tokens": processed['positive_tokens'],
        "processed_negative_tokens": processed['negative_tokens'],
        "mapping_applied": True
    }