# backend/code/disease/disease_vector/verify_vectors.py
# 디렉토리: backend/code/disease/disease_vector

import os
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import logging

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorVerifier:
    def __init__(self, db_url: str):
        self.engine = create_engine(db_url)
        
    def verify_basic_stats(self):
        """기본 통계 확인"""
        print("=== 기본 통계 확인 ===")
        
        query = """
        SELECT 
            (SELECT COUNT(*) FROM tfidf_metadata) as metadata_count,
            (SELECT COUNT(*) FROM disease_vectors) as vector_count,
            (SELECT COUNT(*) FROM vectorization_logs) as log_count;
        """
        
        result = pd.read_sql(query, self.engine)
        print(f"메타데이터 테이블: {result.iloc[0]['metadata_count']}개")
        print(f"벡터 테이블: {result.iloc[0]['vector_count']}개")
        print(f"로그 테이블: {result.iloc[0]['log_count']}개")
        print()
        
    def verify_metadata(self):
        """메타데이터 정보 확인"""
        print("=== 메타데이터 정보 ===")
        
        query = """
        SELECT id, feature_count, min_df, max_df, max_features, 
               ngram_range_min, ngram_range_max, description, created_at
        FROM tfidf_metadata 
        ORDER BY id DESC LIMIT 1;
        """
        
        result = pd.read_sql(query, self.engine)
        for _, row in result.iterrows():
            print(f"메타데이터 ID: {row['id']}")
            print(f"어휘 사전 크기: {row['feature_count']}")
            print(f"최소 문서 빈도: {row['min_df']}")
            print(f"최대 문서 빈도: {row['max_df']}")
            print(f"최대 특성 수: {row['max_features']}")
            print(f"N-gram 범위: {row['ngram_range_min']}-{row['ngram_range_max']}")
            print(f"설명: {row['description']}")
            print(f"생성일: {row['created_at']}")
        print()
        
    def verify_vector_quality(self):
        """벡터 품질 확인"""
        print("=== 벡터 품질 통계 ===")
        
        query = """
        SELECT 
            COUNT(*) as total_vectors,
            MIN(vector_norm) as min_norm,
            MAX(vector_norm) as max_norm,
            AVG(vector_norm) as avg_norm,
            STDDEV(vector_norm) as std_norm,
            MIN(non_zero_count) as min_features,
            MAX(non_zero_count) as max_features,
            AVG(non_zero_count) as avg_features
        FROM disease_vectors;
        """
        
        result = pd.read_sql(query, self.engine)
        for _, row in result.iterrows():
            print(f"전체 벡터 수: {row['total_vectors']}")
            print(f"벡터 노름 - 최소: {row['min_norm']:.4f}, 최대: {row['max_norm']:.4f}, 평균: {row['avg_norm']:.4f}")
            print(f"0이 아닌 특성 수 - 최소: {row['min_features']}, 최대: {row['max_features']}, 평균: {row['avg_features']:.1f}")
        print()
        
    def sample_vectors(self, n=5):
        """샘플 벡터 확인"""
        print(f"=== 샘플 벡터 확인 (상위 {n}개) ===")
        
        query = f"""
        SELECT disease_id, disease_name_ko, department, 
               vector_norm, non_zero_count
        FROM disease_vectors 
        ORDER BY disease_id 
        LIMIT {n};
        """
        
        result = pd.read_sql(query, self.engine)
        for _, row in result.iterrows():
            print(f"ID: {row['disease_id']}")
            print(f"질병명: {row['disease_name_ko']}")
            print(f"진료과: {row['department']}")
            print(f"벡터 노름: {row['vector_norm']:.4f}")
            print(f"특성 수: {row['non_zero_count']}")
            print("-" * 50)
        print()
        
    def test_similarity_calculation(self):
        """유사도 계산 테스트"""
        print("=== 유사도 계산 테스트 ===")
        
        # 임의의 두 질병 벡터 가져오기
        query = """
        SELECT disease_id, disease_name_ko, tfidf_vector
        FROM disease_vectors 
        WHERE vector_norm > 0
        ORDER BY disease_id 
        LIMIT 3;
        """
        
        result = pd.read_sql(query, self.engine)
        
        vectors = []
        disease_names = []
        
        for _, row in result.iterrows():
            # JSON에서 벡터 복원
            vector_data = row['tfidf_vector']
            
            # 이미 dict 형태인지 확인
            if isinstance(vector_data, dict):
                sparse_vector = vector_data
            else:
                # 문자열인 경우 JSON 파싱
                sparse_vector = json.loads(vector_data)
            
            # 5000차원 벡터로 복원 (0으로 초기화)
            full_vector = np.zeros(5000)
            for idx, value in sparse_vector.items():
                full_vector[int(idx)] = value
                
            vectors.append(full_vector)
            disease_names.append(row['disease_name_ko'])
        
        # 코사인 유사도 계산
        if len(vectors) >= 2:
            similarity_matrix = cosine_similarity(vectors)
            
            print("질병 간 코사인 유사도:")
            for i, name1 in enumerate(disease_names):
                for j, name2 in enumerate(disease_names):
                    if i != j:
                        print(f"{name1} vs {name2}: {similarity_matrix[i][j]:.4f}")
        print()
        
    def verify_vocabulary_sample(self):
        """어휘 사전 샘플 확인"""
        print("=== 어휘 사전 샘플 ===")
        
        query = "SELECT vocabulary FROM tfidf_metadata ORDER BY id DESC LIMIT 1;"
        result = pd.read_sql(query, self.engine)
        
        if not result.empty:
            vocab_data = result.iloc[0]['vocabulary']
            
            # 이미 dict 형태인지 확인
            if isinstance(vocab_data, dict):
                vocabulary = vocab_data
            else:
                # 문자열인 경우 JSON 파싱
                vocabulary = json.loads(vocab_data)
            
            # 샘플 단어 10개 표시
            sample_words = list(vocabulary.items())[:10]
            print("어휘 사전 샘플 (단어: 인덱스):")
            for word, idx in sample_words:
                print(f"  {word}: {idx}")
            
            print(f"\n총 어휘 수: {len(vocabulary)}")
        print()
        
    def check_department_distribution(self):
        """진료과별 분포 확인"""
        print("=== 진료과별 질병 분포 ===")
        
        query = """
        SELECT department, COUNT(*) as disease_count
        FROM disease_vectors 
        WHERE department IS NOT NULL
        GROUP BY department 
        ORDER BY disease_count DESC 
        LIMIT 10;
        """
        
        result = pd.read_sql(query, self.engine)
        for _, row in result.iterrows():
            print(f"{row['department']}: {row['disease_count']}개")
        print()
        
    def run_all_verifications(self):
        """모든 검증 실행"""
        print("🔍 TF-IDF 벡터화 결과 검증을 시작합니다...\n")
        
        try:
            self.verify_basic_stats()
            self.verify_metadata()
            self.verify_vector_quality()
            self.sample_vectors()
            self.verify_vocabulary_sample()
            self.check_department_distribution()
            self.test_similarity_calculation()
            
            print("✅ 모든 검증이 완료되었습니다!")
            print("벡터화가 정상적으로 수행되었습니다.")
            
        except Exception as e:
            print(f"❌ 검증 중 오류 발생: {e}")
            logger.error(f"검증 실패: {e}")


def main():
    """메인 실행 함수"""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL 환경변수가 설정되지 않았습니다.")
    
    verifier = VectorVerifier(database_url)
    verifier.run_all_verifications()


if __name__ == "__main__":
    main()