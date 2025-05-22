# disease/disease_vector/create_tfidf_vectors.py
# 디렉토리: backend/code/disease/disease_vector

"""
PostgreSQL TEXT[] 배열을 올바르게 처리하는 TF-IDF 벡터화 스크립트
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DB 연결
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

def parse_postgresql_array(pg_array):
    """
    PostgreSQL TEXT[] 배열에서 Python 리스트 문자열 파싱
    실제 형태: ["['냉방', '생리통', '두통', ...]"]
    """
    try:
        if isinstance(pg_array, list) and len(pg_array) == 1:
            # 첫 번째 원소가 전체 리스트 문자열
            list_string = pg_array[0]
            
            logger.debug(f"파싱할 문자열: {list_string[:100]}...")
            
            # ast.literal_eval로 안전하게 파싱
            try:
                import ast
                parsed_list = ast.literal_eval(list_string)
                
                if isinstance(parsed_list, list):
                    # 유효한 토큰만 필터링
                    clean_tokens = []
                    for token in parsed_list:
                        clean_token = str(token).strip()
                        if len(clean_token) >= 2:  # 2글자 이상
                            clean_tokens.append(clean_token)
                    
                    logger.debug(f"파싱 성공: {len(clean_tokens)}개 토큰")
                    return clean_tokens
                
            except (ValueError, SyntaxError) as e:
                logger.warning(f"ast.literal_eval 실패: {e}")
                
                # ast 실패 시 정규식으로 백업
                import re
                # 작은따옴표로 감싸진 모든 문자열 추출
                pattern = r"'([^']+)'"
                matches = re.findall(pattern, list_string)
                
                if matches:
                    clean_tokens = [match.strip() for match in matches if len(match.strip()) >= 2]
                    logger.debug(f"정규식 파싱 성공: {len(clean_tokens)}개 토큰")
                    return clean_tokens
        
        # 다른 형태의 경우
        elif isinstance(pg_array, list):
            # 일반적인 리스트
            return [str(item).strip() for item in pg_array if len(str(item).strip()) >= 2]
        
        return []
        
    except Exception as e:
        logger.error(f"PostgreSQL 배열 파싱 실패: {str(e)}")
        return []

def load_and_prepare_simple():
    """
    PostgreSQL TEXT[] 배열을 올바르게 처리하여 문서 준비
    """
    try:
        logger.info("PostgreSQL TEXT[] 배열 처리 시작")
        
        # 기존 데이터 삭제
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM disease_vectors"))
            conn.execute(text("DELETE FROM tfidf_metadata"))
            logger.info("기존 데이터 삭제 완료")
        
        # 질병 데이터 로드
        df = pd.read_sql("""
            SELECT id, disnm_ko, disnm_en, dep, tokens
            FROM disv2 
            WHERE tokens IS NOT NULL 
            AND array_length(tokens, 1) > 0
            ORDER BY id
        """, engine)
        
        logger.info(f"로드된 데이터: {len(df)}개")
        
        # 첫 번째 행의 토큰 타입 확인
        if len(df) > 0:
            sample_tokens = df.iloc[0]['tokens']
            logger.info(f"샘플 토큰 타입: {type(sample_tokens)}")
            logger.info(f"샘플 토큰 내용: {sample_tokens}")
        
        # 문서 준비
        documents = []
        disease_info = []
        
        for idx, row in df.iterrows():
            try:
                # PostgreSQL TEXT[] 배열 파싱
                raw_tokens = row['tokens']
                tokens = parse_postgresql_array(raw_tokens)
                
                # 유효한 토큰만 필터링
                clean_tokens = []
                for token in tokens:
                    clean_token = str(token).strip()
                    # 2글자 이상, 의미있는 토큰만
                    if (len(clean_token) >= 2 and 
                        not clean_token.isspace() and
                        clean_token.lower() not in ['none', 'null', '', 'nan']):
                        clean_tokens.append(clean_token)
                
                if len(clean_tokens) == 0:
                    continue
                
                # 문서 생성
                document = " ".join(clean_tokens)
                documents.append(document)
                
                disease_info.append({
                    'id': row['id'],
                    'disease_name_ko': row['disnm_ko'],
                    'disease_name_en': row['disnm_en'],
                    'department': row['dep'],
                    'tokens': clean_tokens
                })
                
                # 처음 3개 샘플 상세 로깅
                if len(documents) <= 3:
                    logger.info(f"=== 샘플 {len(documents)}: {row['disnm_ko']} ===")
                    logger.info(f"  원본 배열: {raw_tokens}")
                    logger.info(f"  파싱된 토큰: {tokens[:10]}")
                    logger.info(f"  정리된 토큰: {clean_tokens[:10]}")
                    logger.info(f"  최종 문서: {document[:100]}...")
                
                if len(documents) % 500 == 0:
                    logger.info(f"문서 처리 진행: {len(documents)}개")
                
            except Exception as e:
                logger.warning(f"행 처리 실패 (ID: {row['id']}): {str(e)}")
                continue
        
        logger.info(f"문서 준비 완료: {len(documents)}개")
        
        # 전체 샘플 확인
        if documents:
            avg_tokens = sum(len(doc.split()) for doc in documents) / len(documents)
            logger.info(f"문서당 평균 토큰 수: {avg_tokens:.1f}")
            
            # 첫 문서의 토큰들 확인
            first_tokens = documents[0].split()
            logger.info(f"첫 문서 토큰 샘플: {first_tokens[:15]}")
            
            # === 추가 디버깅 ===
            logger.info("=== 문서 내용 디버깅 ===")
            for i in range(min(3, len(documents))):
                logger.info(f"문서 {i+1}: '{documents[i][:200]}...'")
                tokens = documents[i].split()
                logger.info(f"  토큰 수: {len(tokens)}")
                logger.info(f"  샘플 토큰: {tokens[:10]}")
                
                # 한글 토큰 확인
                korean_tokens = [t for t in tokens if any('\uAC00' <= c <= '\uD7A3' for c in t)]
                logger.info(f"  한글 토큰: {korean_tokens[:10]}")
                
                # 영어 토큰 확인  
                english_tokens = [t for t in tokens if t.isalpha() and all(ord(c) < 256 for c in t)]
                logger.info(f"  영어 토큰: {english_tokens[:10]}")
            
            # 전체 토큰 통계
            all_tokens = []
            for doc in documents:
                all_tokens.extend(doc.split())
            
            unique_tokens = set(all_tokens)
            korean_unique = {t for t in unique_tokens if any('\uAC00' <= c <= '\uD7A3' for c in t)}
            english_unique = {t for t in unique_tokens if t.isalpha() and all(ord(c) < 256 for c in t)}
            
            logger.info(f"=== 전체 토큰 통계 ===")
            logger.info(f"총 고유 토큰 수: {len(unique_tokens)}")
            logger.info(f"한글 토큰 수: {len(korean_unique)}")
            logger.info(f"영어 토큰 수: {len(english_unique)}")
            logger.info(f"한글 토큰 샘플: {list(korean_unique)[:20]}")
            logger.info(f"영어 토큰 샘플: {list(english_unique)[:20]}")
        
        return documents, disease_info
        
    except Exception as e:
        logger.error(f"데이터 준비 실패: {str(e)}")
        return [], []

def create_simple_vectorizer():
    """
    한글 토큰에 특화된 TF-IDF 벡터라이저
    """
    return TfidfVectorizer(
        min_df=2,                    # 최소 2개 문서
        max_df=0.8,                  # 최대 80% 문서
        max_features=None,           # 특성 수 제한 없음 (모든 토큰 사용)
        ngram_range=(1, 1),          # 1-gram만
        token_pattern=None,          # 기본 패턴 사용 안함
        tokenizer=lambda text: text.split(),  # 공백으로 분할 (이미 토큰화됨)
        lowercase=False,             # 소문자 변환 안함
        stop_words=None,             # 불용어 처리 안함
        norm='l2',
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=True
    )

def safe_json_loads(data):
    """
    안전한 JSON 로드 함수
    """
    if isinstance(data, str):
        return json.loads(data)
    elif isinstance(data, dict):
        return data
    else:
        # PostgreSQL JSONB 타입일 수 있음
        return dict(data) if hasattr(data, '__iter__') else data

def vectorize_and_save(documents, disease_info):
    """
    벡터화 및 저장
    """
    try:
        logger.info("TF-IDF 벡터화 시작")
        
        # 벡터화
        vectorizer = create_simple_vectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        
        vocab_size = len(vectorizer.vocabulary_)
        logger.info(f"벡터화 완료: vocabulary_size={vocab_size}")
        
        # 어휘사전 샘플 확인
        sample_vocab = dict(list(vectorizer.vocabulary_.items())[:15])
        logger.info(f"어휘사전 샘플: {sample_vocab}")
        
        # 의학용어 확인
        medical_terms = []
        for word in vectorizer.vocabulary_.keys():
            if any(keyword in word for keyword in ['통', '열', '기침', '가래', '구토', '설사']):
                medical_terms.append(word)
        logger.info(f"발견된 의학용어: {medical_terms[:10]}")
        
        # 메타데이터 저장
        with engine.begin() as conn:
            # JSON 저장 전 한글 확인
            sample_vocab_check = dict(list(vectorizer.vocabulary_.items())[:10])
            logger.info(f"저장 전 어휘사전 확인: {sample_vocab_check}")
            
            # 한글 토큰 개수 확인
            korean_count = sum(1 for word in vectorizer.vocabulary_.keys() 
                             if any('\uAC00' <= c <= '\uD7A3' for c in word))
            logger.info(f"저장 전 한글 토큰 수: {korean_count}")
            
            metadata = {
                'vocabulary': {word: int(idx) for word, idx in vectorizer.vocabulary_.items()},
                'idf_weights': {word: float(vectorizer.idf_[idx]) 
                              for word, idx in vectorizer.vocabulary_.items()},
                'feature_count': int(vocab_size),
                'min_df': 2,
                'max_df': 0.8,
                'max_features': None,  # 특성 수 제한 없음
                'ngram_range_min': 1,
                'ngram_range_max': 1,
                'description': 'PostgreSQL 배열 기반 TF-IDF v4.2 - 전체 어휘 사용'
            }
            
            # JSON 문자열 생성 시 한글 보존
            try:
                vocab_json = json.dumps(metadata['vocabulary'], ensure_ascii=False, indent=None)
                idf_json = json.dumps(metadata['idf_weights'], ensure_ascii=False, indent=None)
                
                # JSON 생성 후 한글 확인
                logger.info(f"JSON 문자열 길이: vocab={len(vocab_json)}, idf={len(idf_json)}")
                
                # 테스트: JSON을 다시 파싱해서 확인
                test_vocab = json.loads(vocab_json)
                test_korean_count = sum(1 for word in test_vocab.keys() 
                                      if any('\uAC00' <= c <= '\uD7A3' for c in word))
                logger.info(f"JSON 파싱 후 한글 토큰 수: {test_korean_count}")
                
            except Exception as json_error:
                logger.error(f"JSON 생성 중 오류: {json_error}")
                return False
            
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
            
            # 저장 후 즉시 확인 - 수정된 부분
            saved_metadata = conn.execute(text("SELECT vocabulary FROM tfidf_metadata ORDER BY created_at DESC LIMIT 1")).fetchone()
            if saved_metadata:
                # safe_json_loads 함수 사용
                saved_vocab = safe_json_loads(saved_metadata.vocabulary)
                saved_korean_count = sum(1 for word in saved_vocab.keys() 
                                       if any('\uAC00' <= c <= '\uD7A3' for c in word))
                logger.info(f"DB 저장 후 한글 토큰 수: {saved_korean_count}")
                
                sample_saved = dict(list(saved_vocab.items())[:10])
                logger.info(f"DB 저장 후 어휘사전 샘플: {sample_saved}")
            
            # 메타데이터 ID 조회
            metadata_id = conn.execute(text("SELECT id FROM tfidf_metadata ORDER BY created_at DESC LIMIT 1")).scalar()
            # 메타데이터 ID 조회
            metadata_id = conn.execute(text("SELECT id FROM tfidf_metadata ORDER BY created_at DESC LIMIT 1")).scalar()
        
        # 벡터 저장을 별도 트랜잭션으로 처리
        logger.info("벡터 저장 시작")
        saved_count = 0
        batch_size = 100
        
        # 별도의 연결로 벡터 저장
        with engine.connect() as conn:
            for i, info in enumerate(disease_info):
                try:
                    vector = tfidf_matrix[i].toarray().flatten()
                    non_zero_indices = np.nonzero(vector)[0]
                    sparse_vector = {str(int(idx)): float(vector[idx]) for idx in non_zero_indices}
                    vector_norm = float(np.linalg.norm(vector))
                    
                    if vector_norm > 0:  # 0 벡터 제외
                        # sparse_vector를 JSON 문자열로 변환
                        sparse_vector_json = json.dumps(sparse_vector, ensure_ascii=False)
                        
                        with conn.begin():  # 각 벡터마다 개별 트랜잭션
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
                                'tfidf_vector': sparse_vector_json,
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
        
        # 검증
        verify_results()
        
        return True
        
    except Exception as e:
        logger.error(f"벡터화 실패: {str(e)}")
        import traceback
        logger.error(f"상세 에러: {traceback.format_exc()}")
        return False

def verify_results():
    """
    결과 검증
    """
    try:
        with engine.begin() as conn:
            # 메타데이터 확인
            metadata_result = conn.execute(text("SELECT * FROM tfidf_metadata ORDER BY created_at DESC LIMIT 1")).fetchone()
            
            if metadata_result:
                # safe_json_loads 함수 사용
                vocabulary = safe_json_loads(metadata_result.vocabulary)
                
                logger.info(f"총 어휘 수: {len(vocabulary)}")
                
                # 의학용어 확인
                medical_terms = []
                for word in vocabulary.keys():
                    if any(keyword in word for keyword in ['통', '열', '기침', '가래', '구토', '설사']):
                        medical_terms.append(word)
                
                logger.info(f"발견된 의학용어: {medical_terms[:10]}")
                
                # 일반적인 샘플
                sample_vocab = dict(list(vocabulary.items())[:10])
                logger.info(f"어휘사전 샘플: {sample_vocab}")
            
            # 벡터 통계
            vector_stats = conn.execute(text("""
                SELECT COUNT(*) as total, AVG(non_zero_count) as avg_features
                FROM disease_vectors
            """)).fetchone()
            
            logger.info(f"벡터 통계: 총 {vector_stats.total}개, 평균 특성 {vector_stats.avg_features:.1f}개")
        
    except Exception as e:
        logger.error(f"검증 실패: {str(e)}")

def main():
    """
    메인 실행
    """
    try:
        logger.info("PostgreSQL TEXT[] 배열 기반 TF-IDF 벡터화 시작")
        
        # 1. 데이터 준비
        documents, disease_info = load_and_prepare_simple()
        if not documents:
            logger.error("문서 준비 실패")
            return False
        
        # 2. 벡터화 및 저장
        if not vectorize_and_save(documents, disease_info):
            logger.error("벡터화 실패")
            return False
        
        logger.info("PostgreSQL TEXT[] 배열 기반 TF-IDF 벡터화 완료!")
        return True
        
    except Exception as e:
        logger.error(f"실행 실패: {str(e)}")
        import traceback
        logger.error(f"상세 에러: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ PostgreSQL TEXT[] 배열 기반 TF-IDF 벡터화 완료!")
    else:
        print("❌ TF-IDF 벡터화 실패!")
        sys.exit(1)