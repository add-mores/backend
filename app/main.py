# app/main.py
# 디렉토리: backend/app/

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

# 각 API 라우터 import
from app.api import insert_api     # 입력 API (api/insert)
from app.api import disease_api     # 질병 API (api/disease)
from app.api import medicine_api  # 의약품 API (api/medicine)
from app.api import hospital_api    # 병원 API (api/hospital)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="증상 기반 질병 및 의약품 추천 시스템",
    description="사용자의 증상을 분석하여 질병과 의약품, 병원을 추천하는 통합 API",
    version="1.0.0",
    docs_url="/docs",      # Swagger UI
    redoc_url="/redoc"     # ReDoc UI
)

# CORS 설정 (Next.js와 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # Next.js 개발 서버
        "http://127.0.0.1:3000",
        "http://localhost:3001",    # 예비 포트
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE","OPTIONS"],
    allow_headers=["*"],
)

# 각 API 라우터 등록
app.include_router(
    insert_api.router, 
    tags=["증상 처리"],
    prefix="",  # /api/insert 그대로 사용
)

app.include_router(
    disease_api.router, 
    tags=["질병 추천"],
    prefix="",  # /api/disease 그대로 사용
)

app.include_router(
    medicine_api.router, 
    tags=["의약품 추천"],
    prefix="",  # /api/medicine 그대로 사용
)

app.include_router(
    hospital_api.router, 
    tags=["병원 추천"],
    prefix="",  # /api/hospital 그대로 사용
)

# 루트 엔드포인트
@app.get("/")
async def root():
    """API 서버 상태 확인"""
    return {
        "message": "증상 기반 질병 및 의약품 추천 시스템 API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "증상 처리": "/api/insert",
            "질병 추천": "/api/disease", 
            "의약품 추천": "/api/medicine",
            "병원 추천": "/api/hospital",
            "API 문서": "/docs",
            "ReDoc 문서": "/redoc"
        }
    }

# 전체 시스템 헬스 체크
@app.get("/health")
async def health_check():
    """전체 시스템 헬스 체크"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "symptoms": "active",
            "diseases": "active", 
            "medications": "active",
            "hospitals": "active"
        }
    }

# 전체 API 목록
@app.get("/api")
async def api_list():
    """사용 가능한 모든 API 엔드포인트 목록"""
    return {
        "available_endpoints": [
            {
                "path": "/api/insert",
                "method": "POST", 
                "description": "사용자 증상 입력 및 긍정/부정 세그먼트 분리",
                "input": "{ text: '증상 설명' }",
                "output": "{ original_text, positive, negative }"
            },
            {
                "path": "/api/disease",
                "method": "POST",
                "description": "증상 기반 질병 추천 (상위 5개)",
                "input": "입력 API 결과",
                "output": "{ diseases, departments, disease_names }"
            },
            {
                "path": "/api/medicine", 
                "method": "POST",
                "description": "질병명 기반 의약품 추천",
                "input": "{ disease_names: [...] }",
                "output": "{ medications }"
            },
            {
                "path": "/api/hospital",
                "method": "POST", 
                "description": "진료과 및 위치 기반 병원 추천",
                "input": "{ departments: [...], location: {...} }",
                "output": "{ hospitals }"
            }
        ]
    }

# 애플리케이션 시작 이벤트
@app.on_event("startup")
async def startup_event():
    """서버 시작 시 실행되는 함수"""
    logger.info("🚀 증상 기반 질병 및 의약품 추천 시스템 API 서버가 시작되었습니다.")
    logger.info("📖 API 문서: http://localhost:8000/docs")
    logger.info("🔗 엔드포인트:")
    logger.info("   - 증상 처리: POST /api/insert")
    logger.info("   - 질병 추천: POST /api/disease")
    logger.info("   - 의약품 추천: POST /api/medicine")
    logger.info("   - 병원 추천: POST /api/hospital")
from fastapi.responses import JSONResponse
from datetime import datetime

# 애플리케이션 종료 이벤트  
@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 실행되는 함수"""
    logger.info("🛑 API 서버가 종료됩니다.")

# 예외 처리 (전역)
@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """서버 내부 오류 처리"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "서버에서 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """404 에러 처리"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": f"요청한 경로 '{request.url.path}'를 찾을 수 없습니다.",
            "available_endpoints": ["/api/insert", "/api/disease", "/api/medicine", "/api/hospital"],
            "timestamp": datetime.now().isoformat()
        }
    )


# 개발용 정보 (배포 시 제거)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # 개발 모드
        log_level="info"
    )
