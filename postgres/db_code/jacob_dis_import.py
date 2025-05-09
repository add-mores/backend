import pandas as pd
import psycopg2

# CSV 파일 읽기
df = pd.read_csv("/Users/jacob/code/backend/postgres/db_data/dis_prototype.csv", encoding="utf-8")

df['dis_num'] = df['dis_num'].apply(pd.to_numeric, errors='coerce')

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
    cur.execute("INSERT INTO disease (disnm_ko, disnm_en, category, dep, organ, def, coo, sym, sym_k, lapse, diag, therapy, guide, pvt) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                ,(row["disnm_ko"], row["disnm_en"], row["category"], row["dep"], row["organ"], row["def"], row["coo"], row["sym"], row["sym_k"], row["lapse"], row["diag"], row["therapy"], row["guide"], row["pvt"]))
    

conn.commit()
cur.close()
conn.close()
