import time
from urllib.parse import urljoin, urlparse, parse_qs
import pandas as pd
from bs4 import BeautifulSoup
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# 설정
BASE_LIST_URL = "http://www.snuh.org/health/nMedInfo/nList.do"
DETAIL_BASE   = "http://www.snuh.org/health/nMedInfo/"
TOTAL_PAGES   = 189   # 총 게시물 1889건, 페이지당 10건씩
SORT_TYPE     = "R"

# Selenium 헤드리스 설정
options = Options()
options.add_argument("--headless")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)
wait = WebDriverWait(driver, 10)

records = []
all_sections = set(["진료과", "관련 신체기관"])

# 레이블별 링크/텍스트 파싱

def parse_label_list(soup: BeautifulSoup, label: str) -> str:
    for row in soup.select("div.viewRow"):
        em = row.find('em')
        if em and label in em.get_text(separator=' ', strip=True):
            p_tag = row.find('p')
            if not p_tag:
                return ''
            links = p_tag.find_all('a')
            if links:
                # 링크 텍스트를 ',' 구분자로 합침
                return ','.join(a.get_text(strip=True) for a in links)
            text_all = p_tag.get_text(separator=' ', strip=True)
            text_all = re.sub(r'^' + re.escape(label) + r'\s*', '', text_all)
            parts = [part.strip() for part in re.split(r'[,;\s]+', text_all) if part.strip()]
            return ','.join(parts)
    return ''

# 페이지 루프
for page in range(1, TOTAL_PAGES + 1):
    list_url = f"{BASE_LIST_URL}?pageIndex={page}&sortType={SORT_TYPE}&searchNWord=&searchKey="
    print(f"Fetching page {page}/{TOTAL_PAGES}: {list_url}")
    driver.get(list_url)
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.thumbType04 div.item")))

    items = driver.find_elements(By.CSS_SELECTOR, "div.thumbType04 div.item")
    print(f"  Found {len(items)} items on page {page}")

    # 리스트 메타 수집
    entries = []
    for item in items:
        link_elem = item.find_element(By.CSS_SELECTOR, "div.title a")
        href = link_elem.get_attribute("href")
        medid = parse_qs(urlparse(href).query)["medid"][0]
        title_text = link_elem.text.strip()
        summary = item.find_element(By.TAG_NAME, "p").text.strip()
        detail_url = urljoin(DETAIL_BASE, href)
        entries.append({"medid": medid, "title_text": title_text, "summary": summary, "url": detail_url})

    # 상세 페이지 파싱
    for idx, e in enumerate(entries, start=1):
        print(f"    Processing {idx}/{len(entries)}: medid={e['medid']}")
        driver.get(e['url'])
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div.detailWrap")))
        dsoup = BeautifulSoup(driver.page_source, "lxml")

        # 질병명 및 영문명 분리
        full_title_elem = dsoup.select_one("div.viewTitle h3")
        full_title = full_title_elem.get_text(" ", strip=True) if full_title_elem else e['title_text']
        m = re.search(r"\[(.+?)\]", full_title)
        eng_name = m.group(1) if m else ''
        kor_name = re.sub(r"\s*\[.*?\]", "", full_title).strip()

        rec = {
            "medid": e['medid'],
            "질병명": kor_name,
            "영문명": eng_name,
            "요약": e['summary']
        }

        # 목차 섹션 파싱
        wrap = dsoup.select_one("div.detailWrap")
        if wrap:
            for div in wrap.find_all("div", recursive=False):
                h5 = div.find('h5')
                ptag = div.find('p')
                if h5 and ptag:
                    key = h5.get_text(strip=True)
                    val = ptag.get_text("\n", strip=True)
                    rec[key] = val
                    all_sections.add(key)

        # 진료과 & 관련 신체기관 파싱
        rec['진료과'] = parse_label_list(dsoup, '진료과')
        rec['관련 신체기관'] = parse_label_list(dsoup, '관련 신체기관')
        all_sections.update(['진료과', '관련 신체기관'])

        records.append(rec)
        time.sleep(0.2)

# 크롤링 종료
print(f"Total records fetched: {len(records)}")

driver.quit()

# DataFrame 생성 및 CSV 저장
fixed_cols = ["medid", "질병명", "영문명", "요약", "진료과", "관련 신체기관"]
dynamic_cols = sorted(all_sections - set(fixed_cols))
columns = fixed_cols + dynamic_cols

df = pd.DataFrame(records, columns=columns)
df.to_csv("v2_snuh_all_1889.csv", index=False, encoding="utf-8-sig")
print("✅ 전체 크롤링 완료: v2_snuh_all_1889.csv")

