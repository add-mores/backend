# step1_get_content_ids_selenium.py (onclick 기반 페이지네이션 클릭)

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
import pandas as pd
import time
import os

BASE_URL = "https://www.amc.seoul.kr"
CATEGORY_IDS = [f"C0000{str(i).zfill(2)}" for i in range(1, 21)]

def get_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("user-agent=Mozilla/5.0")
    return webdriver.Chrome(options=chrome_options)

def get_content_ids_selenium(category_id, driver):
    url = f"{BASE_URL}/asan/healthinfo/disease/diseaseList.do?diseaseKindId={category_id}"
    driver.get(url)
    time.sleep(2)

    items = []
    current_page = 1

    while True:
        print(f"   - {category_id} 페이지 {current_page} 수집 중: {driver.current_url}")
        disease_links = driver.find_elements(By.XPATH, '//*[@id="listForm"]/div/div/ul/li/div[2]/strong/a')
        for link in disease_links:
            href = link.get_attribute("href")
            name = link.text.strip()
            if "contentId=" in href:
                content_id = href.split("contentId=")[1]
                items.append({"카테고리ID": category_id, "질환명": name, "contentId": content_id})

        try:
            pagination = driver.find_element(By.CLASS_NAME, "numPagingSec")
            page_links = pagination.find_elements(By.TAG_NAME, "a")

            next_page_found = False
            for link in page_links:
                onclick = link.get_attribute("onclick")
                if onclick and f"fnList({current_page + 1})" in onclick:
                    link.click()
                    current_page += 1
                    next_page_found = True
                    time.sleep(2)
                    break

            if not next_page_found:
                break

        except NoSuchElementException:
            break

    print(f"[{category_id}] 추출된 항목 수: {len(items)}")
    return items

# 전체 실행
driver = get_driver()
all_items = []
for cat_id in CATEGORY_IDS:
    print(f"🔍 카테고리 {cat_id} 처리 중...")
    try:
        items = get_content_ids_selenium(cat_id, driver)
        all_items.extend(items)
    except Exception as e:
        print(f"❌ {cat_id} 오류: {e}")
    time.sleep(1.0)
driver.quit()

# 저장
df = pd.DataFrame(all_items)
df.drop_duplicates(subset="contentId", inplace=True)
df.to_csv("content_ids.csv", index=False, encoding="utf-8-sig")
print("✅ content_ids.csv 저장 완료")
