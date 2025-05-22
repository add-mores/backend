# disease/disease_vector/create_tfidf_vectors.py
# 디렉토리: backend/code/disease/disease_vector

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import insert
import logging
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vectorization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TFIDFVectorizer:
    def __init__(self, db_url: str):
        """TF-IDF 벡터화 클래스 초기화"""
        self.engine = create_engine(db_url)
        self.vectorizer = None
        self.metadata_id = None
        
    def load_disease_data(self) -> pd.DataFrame:
        """disv2 테이블에서 질병 데이터 로드"""
        query = """
        SELECT 
            id,
            disnm_ko,
            disnm_en, 
            dep,
            def as definition,
            symptoms,
            therapy,
            tokens,
            tokens_json,
            def_k
        FROM disv2 
        WHERE tokens IS NOT NULL 
        AND array_length(tokens, 1) > 0
        ORDER BY id;
        """
        
        logger.info("질병 데이터 로딩 중...")
        df = pd.read_sql(query, self.engine)
        logger.info(f"총 {len(df)}개 질병 데이터 로드 완료")
        
        return df
    
    def prepare_documents(self, df: pd.DataFrame) -> Tuple[List[str], List[Dict]]:
        """벡터화를 위한 문서 준비"""
        documents = []
        disease_info = []
        
        for _, row in df.iterrows():
            # tokens 배열을 공백으로 연결하여 문서 생성
            if row['tokens'] and len(row['tokens']) > 0:
                # PostgreSQL 배열을 Python 리스트로 변환 (필요시)
                tokens = row['tokens']
                if isinstance(tokens, str):
                    # 문자열로 저장된 경우 파싱
                    tokens = tokens.strip('{}').split(',')
                    tokens = [token.strip().strip('"') for token in tokens if token.strip()]
                
                document = ' '.join(tokens)
                documents.append(document)
                
                disease_info.append({
                    'id': row['id'],
                    'disease_name_ko': row['disnm_ko'],
                    'disease_name_en': row['disnm_en'],
                    'department': row['dep'],
                    'definition': row['definition'],
                    'symptoms': row['symptoms'],
                    'therapy': row['therapy'],
                    'token_count': len(tokens)
                })
        
        logger.info(f"총 {len(documents)}개 문서 준비 완료")
        return documents, disease_info
    
    def create_tfidf_vectorizer(self, min_df: int = 2, max_df: float = 0.8, 
                               max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """TF-IDF 벡터라이저 생성"""
        self.vectorizer = TfidfVectorizer(
            min_df=min_df,              # 최소 2개 문서에서 등장
            max_df=max_df,              # 80% 이상 문서에서 등장하는 단어 제외
            max_features=max_features,   # 최대 5000개 특성
            ngram_range=ngram_range,    # 1-gram, 2-gram 사용
            token_pattern=r'\S+',       # 공백이 아닌 모든 문자 (이미 토큰화됨)
            lowercase=False,            # 이미 전처리됨
            stop_words=None,            # 이미 불용어 제거됨
            norm='l2',                  # L2 정규화
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True          # 로그 스케일링 적용
        )
        
        logger.info(f"TF-IDF 벡터라이저 생성 완료")
        logger.info(f"설정: min_df={min_df}, max_df={max_df}, max_features={max_features}, ngram_range={ngram_range}")
    
    def fit_vectorizer(self, documents: List[str]) -> np.ndarray:
        """TF-IDF 벡터라이저 학습 및 벡터 생성"""
        logger.info("TF-IDF 벡터라이저 학습 중...")
        
        # 벡터 생성
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        
        # 어휘 사전 정보
        vocab_size = len(self.vectorizer.vocabulary_)
        feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"어휘 사전 크기: {vocab_size}")
        logger.info(f"벡터 차원: {tfidf_matrix.shape}")
        logger.info(f"희소성: {(1 - tfidf_matrix.nnz / tfidf_matrix.size) * 100:.2f}%")
        
        return tfidf_matrix
    
    def save_metadata_to_db(self, min_df: int, max_df: float, max_features: int, 
                           ngram_range: Tuple[int, int], description: str = "질병 데이터 TF-IDF v1.0"):
        """TF-IDF 메타데이터를 데이터베이스에 저장"""
        
        # 어휘 사전 생성 {단어: 인덱스} - numpy 타입을 Python 타입으로 변환
        vocabulary = {word: int(idx) for word, idx in self.vectorizer.vocabulary_.items()}
        
        # IDF 가중치 생성 {인덱스: idf값} - numpy 타입을 Python 타입으로 변환
        idf_weights = {str(idx): float(idf) for idx, idf in enumerate(self.vectorizer.idf_)}
        
        metadata = {
            'vocabulary': json.dumps(vocabulary, ensure_ascii=False),
            'idf_weights': json.dumps(idf_weights, ensure_ascii=False),
            'feature_count': int(len(vocabulary)),
            'min_df': int(min_df),
            'max_df': float(max_df),
            'max_features': int(max_features),
            'ngram_range_min': int(ngram_range[0]),
            'ngram_range_max': int(ngram_range[1]),
            'description': description
        }
        
        # 데이터베이스에 저장
        insert_query = """
        INSERT INTO tfidf_metadata 
        (vocabulary, idf_weights, feature_count, min_df, max_df, max_features, 
         ngram_range_min, ngram_range_max, description)
        VALUES (:vocabulary, :idf_weights, :feature_count, :min_df, 
                :max_df, :max_features, :ngram_range_min, :ngram_range_max, :description)
        RETURNING id;
        """
        
        with self.engine.begin() as conn:
            result = conn.execute(text(insert_query), metadata)
            self.metadata_id = result.fetchone()[0]
        
        logger.info(f"메타데이터 저장 완료 (ID: {self.metadata_id})")
    
    def save_vectors_to_db(self, tfidf_matrix: np.ndarray, disease_info: List[Dict]):
        """질병별 TF-IDF 벡터를 데이터베이스에 저장"""
        logger.info("질병별 벡터 저장 중...")
        
        vectors_data = []
        
        for i, info in enumerate(disease_info):
            # 벡터 추출 (희소 행렬에서)
            vector = tfidf_matrix[i].toarray().flatten()
            
            # 0이 아닌 값만 저장 (메모리 효율성) - numpy 타입을 Python 타입으로 변환
            non_zero_indices = np.nonzero(vector)[0]
            sparse_vector = {str(int(idx)): float(vector[idx]) for idx in non_zero_indices}
            
            # 벡터 노름 계산
            vector_norm = float(np.linalg.norm(vector))
            
            vectors_data.append({
                'disease_id': f"disease_{info['id']}",
                'disease_name_ko': info['disease_name_ko'],
                'disease_name_en': info['disease_name_en'],
                'department': info['department'],
                'tfidf_vector': json.dumps(sparse_vector, ensure_ascii=False),
                'vector_norm': vector_norm,
                'non_zero_count': int(len(non_zero_indices)),
                'metadata_id': self.metadata_id
            })
        
        # 배치로 데이터베이스에 저장
        insert_query = """
        INSERT INTO disease_vectors 
        (disease_id, disease_name_ko, disease_name_en, department, 
         tfidf_vector, vector_norm, non_zero_count, metadata_id)
        VALUES (:disease_id, :disease_name_ko, :disease_name_en, :department,
                :tfidf_vector, :vector_norm, :non_zero_count, :metadata_id)
        ON CONFLICT (disease_id) DO UPDATE SET
            tfidf_vector = EXCLUDED.tfidf_vector,
            vector_norm = EXCLUDED.vector_norm,
            non_zero_count = EXCLUDED.non_zero_count,
            metadata_id = EXCLUDED.metadata_id,
            updated_at = CURRENT_TIMESTAMP;
        """
        
        with self.engine.begin() as conn:
            for data in vectors_data:
                conn.execute(text(insert_query), data)
        
        logger.info(f"{len(vectors_data)}개 질병 벡터 저장 완료")
    
    def log_job_status(self, job_name: str, status: str, total_docs: int = 0, 
                      processed_docs: int = 0, vocab_size: int = 0, error_message: str = None):
        """작업 로그 저장"""
        log_data = {
            'job_name': job_name,
            'status': status,
            'total_documents': total_docs,
            'processed_documents': processed_docs,
            'vocabulary_size': vocab_size,
            'error_message': error_message
        }
        
        # completed_at은 완료된 경우에만 설정
        if status == 'completed':
            log_data['completed_at'] = datetime.now()
            insert_query = """
            INSERT INTO vectorization_logs 
            (job_name, status, total_documents, processed_documents, vocabulary_size, 
             error_message, completed_at)
            VALUES (:job_name, :status, :total_documents, :processed_documents,
                    :vocabulary_size, :error_message, :completed_at);
            """
        else:
            insert_query = """
            INSERT INTO vectorization_logs 
            (job_name, status, total_documents, processed_documents, vocabulary_size, 
             error_message)
            VALUES (:job_name, :status, :total_documents, :processed_documents,
                    :vocabulary_size, :error_message);
            """
        
        with self.engine.begin() as conn:
            conn.execute(text(insert_query), log_data)
    
    def run_vectorization(self, min_df: int = 2, max_df: float = 0.8, 
                         max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """전체 벡터화 프로세스 실행"""
        job_name = f"tfidf_vectorization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info("=== TF-IDF 벡터화 프로세스 시작 ===")
            self.log_job_status(job_name, 'running')
            
            # 1. 데이터 로드
            df = self.load_disease_data()
            total_docs = len(df)
            
            # 2. 문서 준비
            documents, disease_info = self.prepare_documents(df)
            
            # 3. 벡터라이저 생성
            self.create_tfidf_vectorizer(min_df, max_df, max_features, ngram_range)
            
            # 4. 벡터화 실행
            tfidf_matrix = self.fit_vectorizer(documents)
            
            # 5. 메타데이터 저장
            self.save_metadata_to_db(min_df, max_df, max_features, ngram_range)
            
            # 6. 벡터 저장
            self.save_vectors_to_db(tfidf_matrix, disease_info)
            
            # 7. 완료 로그
            vocab_size = len(self.vectorizer.vocabulary_)
            self.log_job_status(job_name, 'completed', total_docs, total_docs, vocab_size)
            
            logger.info("=== TF-IDF 벡터화 프로세스 완료 ===")
            logger.info(f"처리된 질병 수: {total_docs}")
            logger.info(f"어휘 사전 크기: {vocab_size}")
            logger.info(f"메타데이터 ID: {self.metadata_id}")
            
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"벡터화 프로세스 실패: {error_msg}")
            self.log_job_status(job_name, 'failed', error_message=error_msg)
            raise


def main():
    """메인 실행 함수"""
    # 데이터베이스 URL
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL 환경변수가 설정되지 않았습니다.")
    
    # 벡터화 실행
    vectorizer = TFIDFVectorizer(database_url)
    
    # 파라미터 설정 (필요에 따라 조정)
    params = {
        'min_df': 2,           # 최소 2개 문서에서 등장
        'max_df': 0.8,         # 80% 이상 문서에서 등장하는 단어 제외
        'max_features': 5000,  # 최대 5000개 특성
        'ngram_range': (1, 2)  # 1-gram, 2-gram
    }
    
    logger.info(f"벡터화 파라미터: {params}")
    
    try:
        success = vectorizer.run_vectorization(**params)
        if success:
            print("\n✅ TF-IDF 벡터화가 성공적으로 완료되었습니다!")
            print(f"로그 파일: vectorization.log")
        else:
            print("\n❌ 벡터화 프로세스에서 오류가 발생했습니다.")
    
    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")
        logger.error(f"실행 실패: {e}")


if __name__ == "__main__":
    main()