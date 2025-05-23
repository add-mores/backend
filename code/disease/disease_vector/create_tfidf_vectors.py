# backend/code/disease/disease_vector/create_simple_tfidf.py
"""
간단하고 안정적인 TF-IDF 벡터화 스크립트
- tokens 컬럼의 고순도 토큰들 직접 사용
- 음수 문제 해결
- PostgreSQL TEXT[] 배열 처리
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DB 연결
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def parse_tokens_array(tokens_array):
    """PostgreSQL TEXT[] 배열을 파싱하여 토큰 리스트 반환"""
    try:
        if isinstance(tokens_array, list) and len(tokens_array) == 1:
            # PostgreSQL에서 가져온 배열 형태: ["['토큰1', '토큰2', ...]"]
            tokens_str = tokens_array[0]
            
            # Python 리스트 문자열을 실제 리스트로 변환
            import ast
            tokens_list = ast.literal_eval(tokens_str)
            
            # 유효한 토큰만 필터링
            valid_tokens = []
            for token in tokens_list:
                clean_token = str(token).strip()
                if len(clean_token) >= 2:  # 2글자 이상
                    valid_tokens.append(clean_token)
            
            return valid_tokens
            
        elif isinstance(tokens_array, list):
            # 일반적인 리스트 형태
            return [str(token).strip() for token in tokens_array if len(str(token).strip()) >= 2]
        
        return []
        
    except Exception as e:
        logger.warning(f"토큰 배열 파싱 실패: {str(e)}")
        return []

def load_disease_data():
    """질병 데이터 로드 및 전처리"""
    try:
        logger.info("질병 데이터 로드 시작")
        
        # disv2 테이블에서 토큰 데이터 로드
        query = """
        SELECT id, disnm_ko, disnm_en, dep, tokens
        FROM disv2 
        WHERE tokens IS NOT NULL 
        AND array_length(tokens, 1) > 0
        ORDER BY id
        """
        
        df = pd.read_sql(query, engine)
        logger.info(f"로드된 데이터: {len(df)}개")
        
        # tokens 컬럼 처리
        valid_docs = []
        disease_info = []
        
        for idx, row in df.iterrows():
            try:
                # PostgreSQL TEXT[] 배열 파싱
                tokens = parse_tokens_array(row['tokens'])
                
                # 유효한 토큰이 있는 경우만 처리
                if len(tokens) >= 1:
                    # 토큰들을 공백으로 연결하여 문서 생성
                    doc = ' '.join(tokens)
                    
                    valid_docs.append(doc)
                    disease_info.append({
                        'id': row['id'],
                        'disease_name_ko': row['disnm_ko'],
                        'disease_name_en': row['disnm_en'],
                        'department': row['dep'],
                        'original_tokens': tokens,
                        'document': doc
                    })
                
                # 진행상황 로그
                if len(valid_docs) % 500 == 0:
                    logger.info(f"처리 진행: {len(valid_docs)}개")
                    
            except Exception as e:
                logger.warning(f"행 처리 실패 (ID: {row['id']}): {str(e)}")
                continue
        
        logger.info(f"유효한 문서: {len(valid_docs)}개")
        
        # 샘플 확인
        for i in range(min(3, len(valid_docs))):
            info = disease_info[i]
            logger.info(f"샘플 {i+1}: {info['disease_name_ko']}")
            logger.info(f"  토큰들: {info['original_tokens'][:10]}")
            logger.info(f"  문서: '{info['document'][:100]}...'")
        
        return valid_docs, disease_info
        
    except Exception as e:
        logger.error(f"데이터 로드 실패: {str(e)}")
        return [], []

def create_tfidf_matrix(documents):
    """간단한 TF-IDF 매트릭스 생성"""
    try:
        logger.info("TF-IDF 벡터화 시작")
        
        # 기본 설정으로 TfidfVectorizer 생성
        vectorizer = TfidfVectorizer(
            min_df=1,                    # 최소 빈도 (관대하게)
            max_df=0.95,                 # 최대 빈도
            max_features=5000,           # 최대 특성 수
            ngram_range=(1, 1),          # 1-gram만
            token_pattern=r'\S+',        # 공백이 아닌 모든 문자
            lowercase=False,             # 소문자 변환 안함
            stop_words=None,             # 불용어 처리 안함
            norm='l2',                   # L2 정규화
            use_idf=True,                # IDF 사용
            smooth_idf=True,             # IDF 스무딩
            sublinear_tf=False           # Sublinear TF 사용 안함 (음수 방지)
        )
        
        # TF-IDF 매트릭스 생성
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        vocab_size = len(vectorizer.vocabulary_)
        logger.info(f"TF-IDF 생성 완료: {vocab_size}개 특성, {tfidf_matrix.shape[0]}개 문서")
        
        # 어휘사전 샘플 확인
        sample_vocab = dict(list(vectorizer.vocabulary_.items())[:10])
        logger.info(f"어휘사전 샘플: {sample_vocab}")
        
        # 의학용어 확인
        medical_terms = []
        for word in vectorizer.vocabulary_.keys():
            if any(keyword in word for keyword in ['통', '열', '기침', '구토', '설사', '복통', '두통']):
                medical_terms.append(word)
        
        logger.info(f"발견된 의학용어: {medical_terms[:15]}")
        
        # 벡터 품질 확인
        logger.info(f"TF-IDF 매트릭스 shape: {tfidf_matrix.shape}")
        logger.info(f"TF-IDF 매트릭스 타입: {type(tfidf_matrix)}")
        logger.info(f"평균 non-zero 개수: {tfidf_matrix.nnz / tfidf_matrix.shape[0]:.1f}")
        
        return vectorizer, tfidf_matrix
        
    except Exception as e:
        logger.error(f"TF-IDF 생성 실패: {str(e)}")
        return None, None

def test_sample_similarity(vectorizer, tfidf_matrix, documents, disease_info):
    """샘플 유사도 테스트"""
    try:
        logger.info("샘플 유사도 테스트 시작")
        
        # 테스트 쿼리들
        test_queries = [
            "두통 편두통 머리",
            "복통 배 아프다",
            "기침 가래 감기",
            "발열 열 고열"
        ]
        
        for query in test_queries:
            logger.info(f"\n테스트 쿼리: '{query}'")
            
            # 쿼리 벡터화
            query_vector = vectorizer.transform([query])
            
            # 모든 질병과 유사도 계산
            similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
            
            # 상위 3개 결과
            top_indices = similarities.argsort()[-3:][::-1]
            
            for idx in top_indices:
                if similarities[idx] > 0:
                    disease_name = disease_info[idx]['disease_name_ko']
                    score = similarities[idx]
                    logger.info(f"  {disease_name}: {score:.4f}")
            
            logger.info(f"  최고 점수: {similarities.max():.4f}")
            logger.info(f"  양수 개수: {(similarities > 0).sum()}")
            
        return True
        
    except Exception as e:
        logger.error(f"샘플 테스트 실패: {str(e)}")
        return False

def save_to_database(vectorizer, tfidf_matrix, disease_info):
    """데이터베이스에 저장"""
    try:
        logger.info("데이터베이스 저장 시작")
        
        # 기존 데이터 삭제
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM disease_vectors"))
            conn.execute(text("DELETE FROM tfidf_metadata"))
            logger.info("기존 데이터 삭제 완료")
        
        # 메타데이터 저장
        vocabulary = {word: int(idx) for word, idx in vectorizer.vocabulary_.items()}
        idf_weights = {word: float(vectorizer.idf_[idx]) for word, idx in vectorizer.vocabulary_.items()}
        
        metadata = {
            'vocabulary': vocabulary,
            'idf_weights': idf_weights,
            'feature_count': len(vocabulary),
            'min_df': 1,
            'max_df': 0.95,
            'max_features': 5000,
            'ngram_range_min': 1,
            'ngram_range_max': 1,
            'description': 'Simple TF-IDF v1.0 - tokens 컬럼 직접 사용'
        }
        
        with engine.begin() as conn:
            # JSON 문자열로 변환 (한글 보존)
            vocab_json = json.dumps(vocabulary, ensure_ascii=False)
            idf_json = json.dumps(idf_weights, ensure_ascii=False)
            
            conn.execute(text("""
                INSERT INTO tfidf_metadata (
                    vocabulary, idf_weights, feature_count, 
                    min_df, max_df, max_features,
                    ngram_range_min, ngram_range_max, description
                ) VALUES (
                    CAST(:vocabulary AS jsonb), CAST(:idf_weights AS jsonb), :feature_count,
                    :min_df, :max_df, :max_features,
                    :ngram_range_min, :ngram_range_max, :description
                )
            """), {
                'vocabulary': vocab_json,
                'idf_weights': idf_json,
                'feature_count': metadata['feature_count'],
                'min_df': metadata['min_df'],
                'max_df': metadata['max_df'],
                'max_features': metadata['max_features'],
                'ngram_range_min': metadata['ngram_range_min'],
                'ngram_range_max': metadata['ngram_range_max'],
                'description': metadata['description']
            })
            
            logger.info("메타데이터 저장 완료")
            
            # 메타데이터 ID 조회
            metadata_id = conn.execute(text("SELECT id FROM tfidf_metadata ORDER BY created_at DESC LIMIT 1")).scalar()
        
        # 벡터 저장
        logger.info("벡터 저장 시작")
        saved_count = 0
        
        with engine.connect() as conn:
            for i, info in enumerate(disease_info):
                try:
                    # Dense 벡터 추출
                    vector = tfidf_matrix[i].toarray().flatten()
                    
                    # 0이 아닌 인덱스만 저장 (sparse 형태)
                    non_zero_indices = np.nonzero(vector)[0]
                    sparse_vector = {str(int(idx)): float(vector[idx]) for idx in non_zero_indices}
                    
                    # 벡터 norm 계산
                    vector_norm = float(np.linalg.norm(vector))
                    
                    if vector_norm > 0:  # 유효한 벡터만 저장
                        with conn.begin():
                            conn.execute(text("""
                                INSERT INTO disease_vectors (
                                    disease_id, disease_name_ko, disease_name_en, 
                                    department, tfidf_vector, vector_norm, 
                                    non_zero_count, metadata_id
                                ) VALUES (
                                    :disease_id, :disease_name_ko, :disease_name_en,
                                    :department, CAST(:tfidf_vector AS jsonb), :vector_norm,
                                    :non_zero_count, :metadata_id
                                )
                            """), {
                                'disease_id': f"disease_{info['id']}",
                                'disease_name_ko': info['disease_name_ko'],
                                'disease_name_en': info['disease_name_en'],
                                'department': info['department'],
                                'tfidf_vector': json.dumps(sparse_vector, ensure_ascii=False),
                                'vector_norm': vector_norm,
                                'non_zero_count': len(non_zero_indices),
                                'metadata_id': metadata_id
                            })
                        saved_count += 1
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"벡터 저장 진행: {i + 1}/{len(disease_info)}")
                        
                except Exception as e:
                    logger.warning(f"벡터 저장 실패 (ID: {info['id']}): {str(e)}")
                    continue
        
        logger.info(f"벡터 저장 완료: {saved_count}개")
        return True
        
    except Exception as e:
        logger.error(f"데이터베이스 저장 실패: {str(e)}")
        return False

def verify_results():
    """결과 검증"""
    try:
        logger.info("결과 검증 시작")
        
        with engine.begin() as conn:
            # 메타데이터 확인
            metadata = conn.execute(text("""
                SELECT feature_count, description, created_at
                FROM tfidf_metadata 
                ORDER BY created_at DESC LIMIT 1
            """)).fetchone()
            
            if metadata:
                logger.info(f"메타데이터: {metadata.feature_count}개 특성, {metadata.description}")
            
            # 벡터 통계
            stats = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_vectors,
                    AVG(non_zero_count) as avg_features,
                    MIN(vector_norm) as min_norm,
                    MAX(vector_norm) as max_norm
                FROM disease_vectors
            """)).fetchone()
            
            if stats:
                logger.info(f"벡터 통계: {stats.total_vectors}개, 평균 특성 {stats.avg_features:.1f}개")
                logger.info(f"Norm 범위: {stats.min_norm:.4f} ~ {stats.max_norm:.4f}")
            
            # 편두통 관련 질병 확인
            headache_diseases = conn.execute(text("""
                SELECT disease_name_ko, non_zero_count, vector_norm
                FROM disease_vectors 
                WHERE disease_name_ko LIKE '%두통%' OR disease_name_ko LIKE '%편두통%'
                LIMIT 5
            """)).fetchall()
            
            logger.info("두통 관련 질병:")
            for disease in headache_diseases:
                logger.info(f"  {disease.disease_name_ko}: {disease.non_zero_count}개 특성, norm={disease.vector_norm:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"검증 실패: {str(e)}")
        return False

def main():
    """메인 실행 함수"""
    try:
        logger.info("=== 간단한 TF-IDF 벡터화 시작 ===")
        
        # 1. 데이터 로드
        documents, disease_info = load_disease_data()
        if not documents:
            logger.error("데이터 로드 실패")
            return False
        
        # 2. TF-IDF 생성
        vectorizer, tfidf_matrix = create_tfidf_matrix(documents)
        if vectorizer is None:
            logger.error("TF-IDF 생성 실패")
            return False
        
        # 3. 샘플 테스트
        if not test_sample_similarity(vectorizer, tfidf_matrix, documents, disease_info):
            logger.error("샘플 테스트 실패")
            return False
        
        # 4. 데이터베이스 저장
        if not save_to_database(vectorizer, tfidf_matrix, disease_info):
            logger.error("데이터베이스 저장 실패")
            return False
        
        # 5. 결과 검증
        if not verify_results():
            logger.error("결과 검증 실패")
            return False
        
        logger.info("=== TF-IDF 벡터화 완료 ===")
        return True
        
    except Exception as e:
        logger.error(f"실행 실패: {str(e)}")
        import traceback
        logger.error(f"상세 에러: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ 간단한 TF-IDF 벡터화 완료!")
    else:
        print("❌ TF-IDF 벡터화 실패!")
        sys.exit(1)