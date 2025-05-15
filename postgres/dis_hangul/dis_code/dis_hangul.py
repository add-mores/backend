import os
import re
import pandas as pd
from konlpy.tag import Okt
from sqlalchemy import create_engine, text
from sqlalchemy.dialects.postgresql import TEXT, JSONB
from dotenv import load_dotenv

# 1) 환경 변수 로드
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL이 설정되어 있지 않습니다")

# 2) DB 연결
engine = create_engine(DATABASE_URL)

# 3) 이전 결과 테이블 제거
with engine.begin() as conn:
    conn.execute(text("DROP TABLE IF EXISTS testdis;"))

# 4) 원본 데이터 읽기
df_raw = pd.read_sql(
    """
    SELECT
      disnm_ko,
      disnm_en,
      dep,
      "def"    AS definition,
      sym      AS symptoms,
      therapy
    FROM disease;
    """,
    engine
)

# 5) 형태소 분석기 초기화
okt = Okt()

# 6) 화이트리스트 (영문·숫자 조합 토큰도 여기에)
WHITELIST = {
    "X-ray","CT","MRI","US","Ultrasound","PET","Endoscopy","Colonoscopy",
    "ECG","EKG","EEG","BP","HR","CBC","BUN","Antibiotic","Rh","RhD",
    "COVID-19","BRCA","BRCA1","BRCA2","XY","DNA","ARC","Cramp","ABO",
    "ASO","ATP","CA125","CA19-9","CEA","Haw River","PSA","G스캐닝",
    "ICL","LCP","MTX","X선","MERRF","N95","XXX","XYY","OTC","PP2","RS",
    "VDT","WPW","A형","B형","O형","C형","D형","G형",
    "18번", "A유형", "B유형", "C유형", "D유형", "G유형",
    "항체A", "항체B", "항체O", "항체D", "항체G",
    "사랑니", "담낭염", "담석증", "간경화", "췌장염"  # 흔한 의학 용어 추가
}

# 7) 불용어 목록 (의학 텍스트에서 유의미하지 않은 단어들)
STOPWORDS = {
    "환자", "진단", "치료", "증상", "질환", "질병", "경우", "이상", 
    "가능", "상태", "병변", "검사", "확인", "종류", "원인", "방법",
    "것", "등", "수", "시", "날", "말", "때", "중", "내", "거", "집",
    "앞", "뒤", "위", "아래", "옆", "년", "월", "일", "시간", "분", "초",
    "이유", "측면", "현대", "음식", "섭취", "크기", "실제", "개인", "일반",
    "관리", "측면", "현대인", "공간", "대구", "모두", "이", "그", "저", "나", 
    "너", "우리", "당신", "자신", "누구", "무엇", "어디", "언제", "어떻게", 
    "어느", "왜", "얼마나", "얼마", "많이", "적게", "더", "덜", "만큼", "정도"
}

# 8) 복합 명사 (형태소 분석에서 쪼개질 수 있는 의미 있는 복합 명사)
COMPOUND_NOUNS = {
    "담낭결석", "췌장염", "장결핵", "충수염", "위궤양", "십이지장궤양", 
    "간경화", "간염", "신부전", "폐렴", "기관지염", "당뇨병", "갑상선염",
    "고혈압", "저혈압", "고지혈증", "심근경색", "뇌졸중", "치매", "골다공증",
    "관절염", "류마티스", "천식", "폐결핵", "백혈병", "빈혈", "대장암", "유방암"
}

# 9) 텍스트 클리닝 함수
def clean_text(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    # 괄호 안 영어·숫자 제거
    txt = re.sub(r"\([^가-힣]*\)", "", txt)
    # ml, %, 단독 숫자 제거 (한글과 붙은 경우는 보존)
    txt = re.sub(r"(?i)\b\d+ml?%?\b", "", txt)
    txt = re.sub(r"\b\d+\b", "", txt)
    # 특수문자를 공백으로 변환
    txt = re.sub(r"[^\w가-힣\s]", " ", txt)
    # 연속 공백 → 단일
    return re.sub(r"\s+", " ", txt).strip()

# 10) 특수 패턴 추출 함수 (명사 형태로 유지)
def extract_special_patterns(text):
    special_tokens = []
    
    # "A형", "B유형" 등을 그대로 추출
    types_regex = re.compile(r"[A-Z][유형]|[A-Z][형]")
    for match in types_regex.finditer(text):
        token = match.group(0)
        if token not in special_tokens:
            special_tokens.append(token)
    
    # "항체 A", "항원 B" 등의 패턴 추출 및 정규화
    reverse_types_regex = re.compile(r"(항체|항원|인자)\s*([A-Z])")
    for match in reverse_types_regex.finditer(text):
        token_type = match.group(1)
        letter = match.group(2)
        token = f"{token_type}{letter}"  # "항체A" 형식으로 정규화
        if token not in special_tokens:
            special_tokens.append(token)
    
    # 한글+형/유형/항체/항/군 패턴 추출
    korean_patterns = [
        r"([가-힣]+형)\b",
        r"([가-힣]+유형)\b",
        r"([가-힣]+항체)\b",
        r"([가-힣]+항원)\b",
        r"([가-힣]+항)\b",
        r"([가-힣]+군)\b",
        r"([가-힣]+인자)\b"
    ]
    
    for pattern in korean_patterns:
        for match in re.finditer(pattern, text):
            token = match.group(1)
            if token not in special_tokens and len(token) >= 2:
                special_tokens.append(token)
    
    return special_tokens

# 11) 한글 명사 토큰 유효성 검사 (개선됨)
def is_valid_token(tok: str) -> bool:
    # 화이트리스트 우선
    if tok in WHITELIST:
        return True
    # 복합 명사 우선
    if tok in COMPOUND_NOUNS:
        return True
    # 불용어 제외
    if tok in STOPWORDS:
        return False
    # 최소 길이 검사 (의미 있는 단어는 보통 2자 이상)
    if len(tok) < 2:
        return False
    # 온전히 한글로만 된 토큰 (또는 한글+영문)
    return bool(re.fullmatch(r"[가-힣A-Za-z]+", tok))

# 12) 의학 용어 추출 보조 함수 (Okt가 놓칠 수 있는 용어 추출)
def extract_medical_terms(text):
    medical_terms = []
    
    # 복합 명사 추출
    for term in COMPOUND_NOUNS:
        if term in text and term not in medical_terms:
            medical_terms.append(term)
    
    # 추가적인 의학 용어 패턴 추출 (예: OO증, OO병, OO암 등)
    medical_patterns = [
        r"([가-힣]{1,5}증)\b",  # 당뇨증, 빈혈증 등
        r"([가-힣]{1,5}병)\b",  # 파킨슨병, 알츠하이머병 등
        r"([가-힣]{1,5}염)\b",  # 위염, 담낭염 등
        r"([가-힣]{1,5}암)\b",  # 폐암, 간암 등
        r"([가-힣]{1,5}통)\b",  # 두통, 복통 등
        r"([가-힣]{1,5}막)\b",  # 망막, 점막 등
        r"([가-힣]{1,5}골)\b",  # 두개골, 척추골 등
        r"([가-힣]{1,5}장애)\b" # 발달장애, 식이장애 등
    ]
    
    for pattern in medical_patterns:
        for match in re.finditer(pattern, text):
            term = match.group(1)
            if term not in medical_terms and len(term) >= 2:
                medical_terms.append(term)
    
    return medical_terms

# 13) 전처리 + 토큰화
records = []
for _, row in df_raw.iterrows():
    definition = clean_text(row["definition"])
    symptoms = clean_text(row["symptoms"])
    therapy = clean_text(row["therapy"])
    combined = f"{definition} {symptoms} {therapy}"

    # 13.1) 화이트리스트 토큰 먼저 추출
    whitelist_tokens = []
    for token in WHITELIST:
        if token in combined and token not in whitelist_tokens:
            whitelist_tokens.append(token)
    
    # 13.2) 특수 패턴 토큰 추출
    special_tokens = extract_special_patterns(combined)
    
    # 13.3) 의학 용어 직접 추출
    medical_tokens = extract_medical_terms(combined)
    
    # 13.4) 형태소 분석을 위한 텍스트에서 이미 추출한 토큰 제거
    temp = combined
    for w in whitelist_tokens + special_tokens + medical_tokens:
        # 단어 경계로 제거 (정규식 이스케이프 처리)
        temp = re.sub(rf"\b{re.escape(w)}\b", " ", temp)
    temp = re.sub(r"\s+", " ", temp).strip()

    # 13.5) 오직 명사만 추출
    nouns = okt.nouns(temp)
    
    # 13.6) 최종 토큰 순서대로 중복 없이 합치기
    all_tokens = whitelist_tokens + special_tokens + medical_tokens + nouns
    final_tokens = []
    
    for token in all_tokens:
        if token and token not in final_tokens and is_valid_token(token):
            final_tokens.append(token)

    records.append({
        "disnm_ko":   row["disnm_ko"],
        "disnm_en":   row["disnm_en"],
        "dep":        row["dep"],
        "definition": definition,
        "symptoms":   row["symptoms"],
        "therapy":    row["therapy"],
        "tokens":     final_tokens,
        "doc":        " ".join(final_tokens)
    })

# 14) 중복(doc) 제거
df_proc = pd.DataFrame(records).drop_duplicates(subset=["disnm_ko", "doc"])

# 15) 결과를 testdis 테이블로 적재
df_proc.to_sql(
    "testdis",
    engine,
    if_exists="replace",
    index=False,
    dtype={
        "disnm_ko":   TEXT,
        "disnm_en":   TEXT,
        "dep":        TEXT,
        "definition": TEXT,
        "symptoms":   TEXT,
        "therapy":    TEXT,
        "tokens":     JSONB,
        "doc":        TEXT
    }
)

print("✅ testdis 테이블이 성공적으로 생성되었습니다.")
print(f"🔢 총 {len(df_proc)}개의 질병 데이터가 처리되었습니다.")