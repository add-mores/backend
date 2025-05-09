import pandas as pd
import requests
import os
import concurrent.futures
import logging
import time
import random
from dotenv import load_dotenv
from tqdm import tqdm

# ─────────────── 설정 ───────────────
load_dotenv()
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# ─────────────── 단일 주소 위경도 요청 함수 (재시도 포함 + 디버깅 로그) ───────────────
def get_lat_lon_retry(address, retry=3):
    for attempt in range(retry):
        try:
            time.sleep(random.uniform(0.1, 0.3))  # API 과부하 방지
            url = "https://dapi.kakao.com/v2/local/search/address.json"
            params = {"query": address}
            res = requests.get(url, headers=headers, params=params, timeout=5)

            if res.status_code != 200:
                logging.warning(f"⚠️ [응답코드 {res.status_code}] {address}")
                continue

            res_json = res.json()

            if "documents" not in res_json:
                logging.warning(f"📛 'documents' 키 없음: {address} → 응답: {res_json}")
                continue

            if not res_json["documents"]:
                logging.warning(f"🔍 주소 미매칭: {address} → 응답 있음 but 결과 없음")
                continue

            x = float(res_json["documents"][0]["x"])
            y = float(res_json["documents"][0]["y"])
            return y, x

        except Exception as e:
            logging.error(f"❌ [재시도 {attempt+1}/{retry}] {address} → 에러: {e}")

    logging.error(f"🚫 최종 실패: {address}")
    return None, None

# ─────────────── 병렬 재요청 함수 ───────────────
def parallel_geocode_retry(address_list, max_workers=15):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_lat_lon_retry, addr) for addr in address_list]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            results.append(future.result())
    return results

# ─────────────── 실패 주소 불러오기 및 처리 ───────────────
fail_df = pd.read_csv("../../pages/missing_hospitals.csv")
addresses = fail_df["address"].dropna().unique().tolist()

logging.info(f"🚀 재처리 시작: 실패 주소 {len(addresses)}개")

coords = parallel_geocode_retry(addresses)
fail_df["lat"], fail_df["lon"] = zip(*coords)

# 복구된 주소 저장
recovered = fail_df.dropna(subset=["lat", "lon"])
recovered.to_csv("../../pages/recovered_coordinates.csv", index=False)

# 실패한 것들 별도로 저장
still_failed = fail_df[fail_df["lat"].isna()]
still_failed.to_csv("still_failed.csv", index=False)

# 결과 요약
logging.info(f"✅ 복구 완료: {len(recovered)}개 복구 / {len(fail_df)}개 중")
logging.info(f"❌ 여전히 실패: {len(still_failed)}개 → still_failed.csv 저장")

