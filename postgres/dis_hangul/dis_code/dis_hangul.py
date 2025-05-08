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
    "COVID 19","BRCA","BRCA1","BRCA2","XY","DNA","ARC","Cramp","ABO",
    "ASO","ATP","CA125","CA19-9","CEA","Haw River","PSA","G스캐닝",
    "ICL","LCP","MTX","X선","MERRF","N95","XXX","XYY","OTC","PP2","RS",
    "VDT","WPW","A형","B형","O형","C형","D형","G형",
    "18번"
}

# 7) “형, 유형, 항체, 항, 군” 등의 예외 패턴
EXC_FULL = re.compile(
    r"\b[\w가-힣]*(?:유형|항체|형|항|군)\b"
)

# 8) 텍스트 클리닝 함수
def clean_text(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    # 괄호 안 영어·숫자 제거
    txt = re.sub(r"\([^가-힣]*\)", "", txt)
    # ml, %, 단독 숫자 제거 (한글과 붙은 경우는 보존)
    txt = re.sub(r"(?i)\b\d+ml?%?\b", "", txt)
    txt = re.sub(r"\b\d+\b", "", txt)
    # 특수문자 제거
    txt = re.sub(r"[^\w가-힣\s]", " ", txt)
    # 연속 공백 → 단일
    return re.sub(r"\s+", " ", txt).strip()

# 9) 한글 명사 토큰 유효성 검사
def is_valid_token(tok: str) -> bool:
    # 화이트리스트 우선
    if tok in WHITELIST:
        return True
    # 온전히 한글로만 된 토큰
    return bool(re.fullmatch(r"[가-힣]+", tok))

# 10) 전처리 + 토큰화
records = []
for _, row in df_raw.iterrows():
    definition = clean_text(row["definition"])
    symptoms   = clean_text(row["symptoms"])
    combined   = f"{definition} {symptoms}"

    # 10.1) 화이트리스트 토큰 먼저 추출
    wl = []
    for w in combined.split():
        if w in WHITELIST and w not in wl:
            wl.append(w)

    # 10.2) 예외 패턴 토큰 추출
    excs = EXC_FULL.findall(combined)
    excs = [e.strip() for e in excs if e.strip()]

    # 10.3) 형태소 분석을 위한 텍스트에서 WL·EXC 제거
    temp = combined
    for w in wl + excs:
        # 단어 경계로 제거
        temp = re.sub(rf"\b{re.escape(w)}\b", " ", temp)
    temp = re.sub(r"\s+", " ", temp)

    # 10.4) Okt로 명사만 추출
    nouns = okt.nouns(temp)

    # 10.5) 최종 토큰 순서대로 중복 없이 합치기
    toks = []
    for w in wl + excs + nouns:
        if w and w not in toks and is_valid_token(w):
            toks.append(w)

    records.append({
        "disnm_ko":   row["disnm_ko"],
        "disnm_en":   row["disnm_en"],
        "dep":        row["dep"],
        "definition": definition,
        "symptoms":   row["symptoms"],
        "therapy":    row["therapy"],
        "tokens":     toks,
        "doc":        " ".join(toks)
    })

# 11) 중복(doc) 제거
df_proc = pd.DataFrame(records).drop_duplicates(subset=["disnm_ko", "doc"])

# 12) 결과를 testdis 테이블로 적재
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
        "symptoms":   JSONB,
        "therapy":    TEXT,
        "tokens":     JSONB,
        "doc":        TEXT
    }
)

print("✅ testdis 테이블이 성공적으로 생성되었습니다.")
