# app/models/database.py
# 디렉토리: backend/app/models/

import os
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv
import logging

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)

# 데이터베이스 URL 가져오기
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL 환경변수가 설정되지 않았습니다.")

# SQLAlchemy 엔진 생성
engine = create_engine(
    DATABASE_URL,
    echo=False,  # SQL 쿼리 로깅 (개발시에는 True로 설정 가능)
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # 연결 상태 확인
    pool_recycle=3600    # 1시간마다 연결 재사용
)

# 세션 팩토리 생성
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# 베이스 클래스 (ORM 모델용, 현재는 사용하지 않음)
Base = declarative_base()

# 데이터베이스 세션 의존성 함수
def get_db() -> Session:
    """
    FastAPI 의존성으로 사용할 데이터베이스 세션 생성 함수
    
    사용 예시:
    @router.post("/api/disease")
    async def some_endpoint(db: Session = Depends(get_db)):
        # db 사용
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"데이터베이스 세션 오류: {e}")
        db.rollback()
        raise
    finally:
        db.close()

# 데이터베이스 연결 테스트 함수
def test_connection():
    """데이터베이스 연결 테스트"""
    try:
        with engine.begin() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info("✅ 데이터베이스 연결 성공")
            return True
    except Exception as e:
        logger.error(f"❌ 데이터베이스 연결 실패: {e}")
        return False

# 질병 관련 데이터베이스 함수들
class DatabaseService:
    """데이터베이스 관련 서비스 클래스"""
    
    @staticmethod
    def get_medical_mappings(db: Session) -> dict:
        """의학용어 매핑 데이터 조회"""
        try:
            query = text("SELECT common_term, medical_term FROM medical_term_mappings")
            result = db.execute(query)
            
            mappings = {}
            for row in result:
                mappings[row.common_term] = row.medical_term
            
            logger.info(f"의학용어 매핑 로드: {len(mappings)}개")
            return mappings
        
        except Exception as e:
            logger.error(f"의학용어 매핑 조회 실패: {e}")
            return {}
    
    @staticmethod
    def get_tfidf_metadata(db: Session) -> dict:
        """TF-IDF 메타데이터 조회"""
        try:
            query = text("""
                SELECT vocabulary, idf_weights, feature_count 
                FROM tfidf_metadata 
                ORDER BY id DESC 
                LIMIT 1
            """)
            result = db.execute(query).fetchone()
            
            if not result:
                raise Exception("TF-IDF 메타데이터를 찾을 수 없습니다.")
            
            return {
                'vocabulary': result.vocabulary,
                'idf_weights': result.idf_weights,
                'feature_count': result.feature_count
            }
        
        except Exception as e:
            logger.error(f"TF-IDF 메타데이터 조회 실패: {e}")
            raise
    
    @staticmethod
    def get_disease_vectors(db: Session) -> list:
        """질병 벡터 데이터 조회"""
        try:
            query = text("""
                SELECT 
                    dv.disease_id, 
                    dv.disease_name_ko, 
                    dv.department, 
                    dv.tfidf_vector, 
                    dv.vector_norm,
                    d.def as definition, 
                    d.symptoms, 
                    d.therapy
                FROM disease_vectors dv
                LEFT JOIN disv2 d ON CAST(SUBSTRING(dv.disease_id FROM 9) AS INTEGER) = d.id
                WHERE dv.vector_norm > 0
                ORDER BY dv.disease_id
            """)
            
            result = db.execute(query)
            diseases = []
            
            for row in result:
                diseases.append({
                    'disease_id': row.disease_id,
                    'disease_name': row.disease_name_ko,
                    'department': row.department or '',
                    'definition': row.definition or '',
                    'symptoms': row.symptoms or '',
                    'therapy': row.therapy or '',
                    'tfidf_vector': row.tfidf_vector,
                    'vector_norm': row.vector_norm
                })
            
            logger.info(f"질병 벡터 로드: {len(diseases)}개")
            return diseases
        
        except Exception as e:
            logger.error(f"질병 벡터 조회 실패: {e}")
            raise

# 초기화 시 연결 테스트
if __name__ == "__main__":
    # 직접 실행시 연결 테스트
    print("데이터베이스 연결 테스트 중...")
    if test_connection():
        print("✅ 연결 성공!")
        
        # 추가 테스트
        with SessionLocal() as db:
            try:
                mappings = DatabaseService.get_medical_mappings(db)
                print(f"의학용어 매핑: {len(mappings)}개")
                
                metadata = DatabaseService.get_tfidf_metadata(db)
                print(f"TF-IDF 어휘 수: {metadata['feature_count']}개")
                
                diseases = DatabaseService.get_disease_vectors(db)
                print(f"질병 벡터: {len(diseases)}개")
                
            except Exception as e:
                print(f"❌ 데이터 조회 실패: {e}")
    else:
        print("❌ 연결 실패!")