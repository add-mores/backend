import pandas as pd
import re
import requests
import time
import random
import os
from dotenv import load_dotenv
from tqdm import tqdm
import concurrent.futures

# ───────────── 1. 설정 ─────────────
load_dotenv()
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}

# ───────────── 2. 주소 정제 함수 ─────────────
def clean_address(addr):
    if pd.isna(addr):
        return ""
    addr = str(addr)
    addr = re.sub(r"\s*\([^)]*\)", "", addr)
    if "," in addr:
        addr = addr.split(",")[0].strip()
    return addr.strip()

# ───────────── 3. 카카오 API 요청 함수 ─────────────
def get_lat_lon_retry(address, retry=3):
    for _ in range(retry):
        try:
            time.sleep(random.uniform(0.1, 0.3))
            url = "https://dapi.kakao.com/v2/local/search/address.json"
            params = {"query": address, "analyze_type": "exact"}
            res = requests.get(url, headers=headers, params=params, timeout=5)
            if res.status_code != 200:
                continue
            documents = res.json().get("documents", [])
            if not documents:
                continue
            x = float(documents[0]["x"])
            y = float(documents[0]["y"])
            return y, x
        except:
            continue
    return None, None

# ───────────── 4. 병렬 처리 (순서 보장) ─────────────
def parallel_geocode_ordered(address_list, max_workers=20):
    results = [None] * len(address_list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(get_lat_lon_retry, addr): idx
            for idx, addr in enumerate(address_list)
        }
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(address_list), desc="📍 병렬 위경도 변환"):
            idx = future_to_index[future]
            try:
                results[idx] = future.result()
            except:
                results[idx] = (None, None)
    return results

# ───────────── 5. 메인 실행 ─────────────
if __name__ == "__main__":
    df = pd.read_csv("../../pages/hospital_combined.csv").dropna(subset=["address"]).copy()

    df["cleaning"] = df["address"].apply(clean_address)
    coords = parallel_geocode_ordered(df["cleaning"].tolist(), max_workers=20)
    df["lat"], df["lon"] = zip(*coords)

    # 🔎 6. 매칭 확인용 로그 5개 출력
    print("\n📌 [주소 ↔ 위경도 매칭 확인]")
    print(df[["address", "cleaning", "lat", "lon"]].head(5).to_string(index=False))

    # 7. 성공/실패 분리
    success_df = df[df["lat"].notna() & df["lon"].notna()].copy()
    fail_df = df[df["lat"].isna() | df["lon"].isna()].copy()

    # 8. 저장
    fail_df.to_csv("missing_hospitals_failed.csv", index=False)
    success_df.drop(columns=["cleaning"], inplace=True)
    success_df.to_csv("missing_hospitals_success.csv", index=False)

    # 9. 로그
    print(f"\n✅ 변환 성공: {len(success_df)}건")
    print(f"❌ 변환 실패: {len(fail_df)}건")

