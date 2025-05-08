import pandas as pd
import folium
import requests
from math import radians, sin, cos, sqrt, atan2
from dotenv import load_dotenv
import os

# .env에서 API 키 로드
load_dotenv()
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}

# 주소 → 좌표 변환 함수
def get_lat_lon(address):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    params = {"query": address}
    res = requests.get(url, headers=headers, params=params)
    res_json = res.json()
    if res_json["documents"]:
        x = float(res_json["documents"][0]["x"])
        y = float(res_json["documents"][0]["y"])
        return y, x
    return None, None

# 거리 계산 함수
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # 지구 반지름 (km)
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# 병원 데이터 로드
df = pd.read_csv("../../pages/hospital_filtered_1.csv")

# 위경도 추출 (시간 절약 위해 최초 1회 실행 후 CSV로 저장 권장)
df["lat"], df["lon"] = zip(*df["address"].apply(get_lat_lon))
df.dropna(subset=["lat", "lon"]).to_csv("../../pages/hospital_with_latlon.csv", index=False)
