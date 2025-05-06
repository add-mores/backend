import pandas as pd
import psycopg2

conn = psycopg2.connect(
    dbname='dev_db', user='devuser', password='devpass', host='localhost', port='5432'
)
df = pd.read_sql_query("SELECT * FROM disease3", conn)
df.to_csv("dis_prototype.csv", index=False,  encoding="utf-8-sig")
conn.close()

