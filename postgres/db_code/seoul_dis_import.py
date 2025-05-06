import pandas as pd
import psycopg2

# CSV 파일 읽기
df = pd.read_csv("../db_data/seoul_disease.csv", encoding="utf-8")

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
    cur.execute("INSERT INTO disease1 (disNm_ko, disNm_en, dep, organ, lapse, guide, pvt, coo, def, sym, diag, therapy) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",(row["질병명"], row["영문명"], row["진료과"], row["관련 신체기관"], row["경과/합병증"], row["식이요법/생활가이드"], row["예방방법"], row["원인"], row["정의"], row["증상"], row["진단/검사"], row["치료"])
    )

conn.commit()
cur.close()
conn.close()
