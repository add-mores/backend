import re
import os
import pandas as pd
from konlpy.tag import Okt
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import TEXT
from dotenv import load_dotenv

# 1) 환경 변수 로드 및 DB 연결
load_dotenv()  # .env에서 DATABASE_URL을 로드
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL이 설정되어 있지 않습니다.")
engine = create_engine(DATABASE_URL, echo=True)

# 2) 형태소 분석기 초기화
okt = Okt()

# 3) 영어 약어 화이트리스트
WHITELIST = {
    "X-ray","CT","MRI","US","Ultrasound","PET","Endoscopy","Colonoscopy",
    "ECG","EKG","EEG","BP","HR","CBC","BUN","Antibiotic","Rh","RhD",
    "COVID19","BRCA","BRCA1","BRCA2","XY","DNA","ARC","Cramp","ABO",
    "ASO","ATP","CA125","CA19-9","CEA","HawRiver","PSA","G스캐닝",
    "ICL","LCP","MTX","X선","MERRF","N95","XXX","XYY","OTC","PP2",
    "RS","VDT","WPW"
}

# 4) 한글 예외 패턴: “형”, “항체”, “항”, “군” 포함 토큰은 무조건 보존
EXC_PATTERN = re.compile(r".*(형|항체|항|군).*")

# 5) 텍스트 클리닝 함수
def clean(text: str) -> str:
    text = re.sub(r"\([^가-힣]*[A-Za-z0-9][^)]*\)", " ", str(text))
    text = text.replace("(", " ").replace(")", " ")
    text = re.sub(r"[^가-힣A-Za-z0-9\s\-]", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()

# 6) 토큰화 함수
def tokenize(text: str) -> list[str]:
    try:
        cleaned = clean(text)
        morphs = okt.morphs(cleaned, stem=True)
        tokens = []
        for m in morphs:
            if m in WHITELIST or EXC_PATTERN.match(m) or re.fullmatch(r"[가-힣]{2,}", m):
                tokens.append(m)
        return tokens
    except Exception as e:
        print(f"[Warning] tokenize 실패: {e!r} for text={text!r}")
        return []

# 7) Raw 테이블에서 필요한 컬럼 읽어오기 (테이블명: disease)
df_raw = pd.read_sql(
    """
    SELECT
      disnm_ko,
      disnm_en,
      dep,
      "def"   AS definition,
      sym      AS symptoms,
      therapy
    FROM disease
    """,
    engine
)

# 8) 전처리 및 토큰화
processed = []
for _, row in df_raw.iterrows():
    parts = [row.definition, row.symptoms, row.therapy]
    text_block = " ".join(filter(None, parts))
    toks = tokenize(text_block) or []
    processed.append({
        "name_ko":    row.disnm_ko,
        "name_en":    row.disnm_en,
        "department": row.dep,
        "tokens":     toks,
        "doc":        " ".join(toks)
    })

proc_df = pd.DataFrame(processed)

# 9) 결과를 testdis 테이블에 저장 (append 모드, SERIAL PK 유지)
proc_df.to_sql(
    "testdis",
    con=engine,
    if_exists="append",
    index=False,
    dtype={
        "name_ko":    TEXT,
        "name_en":    TEXT,
        "department": TEXT,
        "tokens":     JSONB,
        "doc":        TEXT
    }
)

print("✅ testdis 테이블에 전처리 결과 적재 완료")





