from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os
import requests
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시엔 ["https://addmore.kr"] 권장
    allow_credentials=True,
    allow_methods=["*"],  # ← 이거 없으면 405 무조건 뜸
    allow_headers=["*"],  # ← 이것도 거의 항상 필요
)

DATABASE_URL = os.getenv("DATABASE_URL")
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
engine = create_engine(DATABASE_URL)

def haversine(lat1, lon1, lat2s, lon2s):
    R = 6371
    dlat = np.radians(lat2s - lat1)
    dlon = np.radians(lon2s - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2s)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

class FilterRequest(BaseModel):
    lat: float
    lon: float
    radius: float = 1.0
    deps: Optional[List[str]] = None
    search_name: Optional[str] = None

@app.post("/api/filter_hospitals")
def filter_hospitals(req: FilterRequest):
    query = """
    SELECT hos_nm, hos_type, pv, city, add, deps, lat, lon
    FROM testhosp
"""
    df = pd.read_sql(query, engine).dropna(subset=["lat", "lon"])
    df["distance"] = haversine(req.lat, req.lon, df["lat"].values, df["lon"].values)

    if req.deps:
        df = df[df["deps"].apply(lambda t: any(dept.strip() in t.split(",") for dept in req.deps if t))]
    if req.search_name:
        df = df[df["hos_nm"].str.contains(req.search_name, case=False, na=False)]

    df = df[df["distance"] <= req.radius].sort_values("distance")
    records = []
    for _, row in df.head(30).iterrows():
        records.append({
            "hos_nm": row["hos_nm"],
            "add": row["add"],
            "deps": row["deps"],
            "lat": row["lat"],
            "lon": row["lon"],
            "distance": round(row["distance"], 2)
        })
    return records

@app.get("/list_departments")
def list_departments():
    df = pd.read_sql("SELECT deps FROM testhosp", engine)

    all_depts = set()
    for deps in df["deps"].dropna():
        for d in deps.split(","):
            all_depts.add(d.strip())

    return sorted(all_depts)

@app.get("/geocode")
def geocode_address(query: str = Query(...)):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    try:
        res = requests.get(url, headers=headers, params={"query": query})
        res.raise_for_status()
        data = res.json()
        docs = data.get("documents")
        if docs:
            return {
                "lat": float(docs[0]["y"]),
                "lon": float(docs[0]["x"]),
                "address_name": docs[0].get("address_name")
            }
        return {"error": f"주소를 찾을 수 없습니다. Kakao 응답: {data}"}
    except Exception as e:
        return {"error": str(e)}


