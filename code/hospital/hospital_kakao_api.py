import pandas as pd
import requests
import os
import concurrent.futures
import logging
import time
import random
from dotenv import load_dotenv
from math import radians, sin, cos, sqrt, atan2
from tqdm import tqdm

# ─────────────── 설정 ───────────────
load_dotenv()
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ─────────────── 위경도 변환 함수 ───────────────
def get_lat_lon(address):
    try:
        time.sleep(random.uniform(0.1, 0.3))  # ✅ 0.1~0.3초 랜덤 딜레이
        url = "https://dapi.kakao.com/v2/local/search/address.json"
        params = {"query": address}
        res = requests.get(url, headers=headers, params=params, timeout=5)
        res.raise_for_status()
        res_json = res.json()
        if res_json["documents"]:
            x = float(res_json["documents"][0]["x"])
            y = float(res_json["documents"][0]["y"])
            return y, x
    except Exception as e:
        logging.error(f"❌ 주소 변환 실패: {address} | {e}")
    return None, None

# ─────────────── 거리 계산 함수 (옵션) ───────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# ─────────────── CSV 읽기 및 병렬 처리 ───────────────
df = pd.read_csv("../../pages/hospital_combined.csv")
addresses = df["address"].tolist()

def parallel_geocode(address_list, max_workers=20):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_lat_lon, addr) for addr in address_list]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.append(future.result())
    return results

# ─────────────── 실행 및 저장 ───────────────
coords = parallel_geocode(addresses)
df["lat"], df["lon"] = zip(*coords)
df.dropna(subset=["lat", "lon"]).to_csv("../../pages/hospital_with_latlon.csv", index=False)

