# step2_parse_diseases_fixed.py (scraped_category 기반 리팩터링)

import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlencode
import time

INPUT_CSV = "content_ids.csv"
OUTPUT_CSV = "asan_disease_data.csv"

BASE_URL = "https://www.amc.seoul.kr/asan/healthinfo/disease/diseaseDetail.do?contentId={}"

# 텍스트 추출 유틸 함수
def extract_dt_dd(soup, dt_name):
    try:
        dt = soup.find("dt", string=dt_name)
        if dt:
            dd = dt.find_next_sibling("dd")
            return dd.get_text(strip=True) if dd else ""
        return ""
    except:
        return ""

def extract_keywords(soup):
    try:
        tag = soup.find("dt", string="증상")
        if not tag:
            return ""
        dd = tag.find_next_sibling("dd")
        if not dd:
            return ""
        return ", ".join([a.get_text(strip=True) for a in dd.find_all("a")])
    except:
        return ""

def parse_detail_page(content_id):
    url = BASE_URL.format(content_id)
    try:
        res = requests.get(url, timeout=5)
        soup = BeautifulSoup(res.text, "html.parser")

        # 기본 정보
        title = soup.select_one("strong.contTitle")
        진료과 = extract_dt_dd(soup, "진료과")
        키워드 = extract_keywords(soup)

        # 본문 항목
        정의 = extract_dt_dd(soup, "정의")
        원인 = extract_dt_dd(soup, "원인")
        증상 = extract_dt_dd(soup, "증상")
        진단 = extract_dt_dd(soup, "진단")
        치료 = extract_dt_dd(soup, "치료")
        경과 = extract_dt_dd(soup, "경과") or extract_dt_dd(soup, "경과/합병증")
        주의사항 = extract_dt_dd(soup, "주의사항")

        return {
            "contentId": content_id,
            "질환명": title.get_text(strip=True) if title else "",
            "진료과": 진료과,
            "증상_Keyword": 키워드,
            "정의": 정의,
            "원인": 원인,
            "증상": 증상,
            "진단": 진단,
            "치료": 치료,
            "경과": 경과,
            "주의사항": 주의사항,
            "출처_URL": url
        }
    except Exception as e:
        print(f"❌ {content_id} 크롤링 실패: {e}")
        return {"contentId": content_id, "출처_URL": url, "크롤링실패": True}

# 메인 실행
content_df = pd.read_csv(INPUT_CSV, dtype={"contentId": str})
results = []

for i, row in content_df.iterrows():
    cid = row["contentId"]
    print(f"[{i+1}/{len(content_df)}] contentId: {cid}")
    result = parse_detail_page(cid)
    results.append(result)
    time.sleep(0.5)

# 저장
final_df = pd.DataFrame(results)
final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print("✅ 질환 상세 크롤링 완료 및 저장됨 → asan_disease_data.csv")