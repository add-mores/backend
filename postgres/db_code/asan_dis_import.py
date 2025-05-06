import pandas as pd
import psycopg2

# CSV 파일 읽기
df = pd.read_csv("../db_data/asan_disease.csv", encoding="utf-8")

# PostgreSQL 연결
conn = psycopg2.connect(
    host="localhost",
    dbname="dev_db",
    user="devuser",
    password="devpass",
)
cur = conn.cursor()

# 각 행을 INSERT
for _, row in df.iterrows():
    cur.execute("INSERT INTO disease2 (disNm_ko, disNm_en, dep, sym_k, lapse, coo, def, sym, diag, therapy) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",(row["질환명_한글"], row["질환명_영문"], row["진료과"], row["증상_Keyword"], row["경과"], row["원인"], row["정의"], row["증상"], row["진단"], row["치료"])
    )

conn.commit()
cur.close()
conn.close()
