import time
import re
from urllib.parse import urljoin, parse_qs, urlparse
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# --- Asan 질환백과 전체 크롤러 (GET 기반 페이지네이션) ---
BASE_LIST_URL = "https://www.amc.seoul.kr/asan/healthinfo/disease/diseaseList.do"
DETAIL_BASE   = "https://www.amc.seoul.kr"
KIND_CODES    = [f"C{str(i).zfill(6)}" for i in range(1, 21)]
OUTPUT_FILE   = "v2_asan_disease_all.csv"
PER_PAGE      = 20

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

for kind in KIND_CODES:
    # 첫 페이지 로드 및 총 페이지 수 계산
    first_url = f"{BASE_LIST_URL}?diseaseKindId={kind}&searchKeyword=&pageIndex=1"
    driver.get(first_url)
    wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.listCont li")))
    soup0 = BeautifulSoup(driver.page_source, 'lxml')
    spans = soup0.select('div.pagingWrapSec .numPagingSec a span')
    page_nums = [int(s.get_text(strip=True)) for s in spans if s.get_text(strip=True).isdigit()]
    total_pages = max(page_nums) if page_nums else 1
    print(f"Category {kind}: total_pages={total_pages}")

    # 페이지별 순회
    for page in range(1, total_pages + 1):
        print(f"  Crawling page {page}/{total_pages}")
        page_url = f"{BASE_LIST_URL}?diseaseKindId={kind}&searchKeyword=&pageIndex={page}"
        driver.get(page_url)
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.listCont li")))
        time.sleep(0.5)

        soup = BeautifulSoup(driver.page_source, 'lxml')
        items = soup.select('div.listCont li')
        print(f"   Found {len(items)} items on page {page}")

        for item in items:
            # 목록에서 제목과 링크 추출
            a = item.select_one('strong.contTitle a')
            if not a:
                continue
            href = a['href']
            detail_url = urljoin(DETAIL_BASE, href)
            full_title = a.get_text(strip=True)

            # 영문명과 한글명 분리: 외부 괄호만 제거, 내부 괄호 유지
            m = re.search(r"[\(\[](.+)[\)\]]\s*$", full_title)
            if m:
                eng = m.group(1)
                kor = full_title[:m.start()].strip()
            else:
                eng = ''
                kor = full_title

            # contentId 추출
            cid = parse_qs(urlparse(href).query).get('contentId', [''])[0]

            rec = {'kind': kind, 'contentId': cid, '질병명': kor, '영문명': eng}

            # 증상_keyword 추출
            sym_dd = item.select_one('dl dt:-soup-contains("증상") + dd')
            if sym_dd:
                links = sym_dd.find_all('a')
                rec['증상_keyword'] = ','.join(l.get_text(strip=True) for l in links) if links else sym_dd.get_text(' ', strip=True)

            # 상세 페이지 크롤링
            driver.get(detail_url)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, 'dl.descDl')))
            dsoup = BeautifulSoup(driver.page_source, 'lxml')
            detail_dl = dsoup.select_one('dl.descDl')
            if detail_dl:
                for dt, dd in zip(detail_dl.find_all('dt'), detail_dl.find_all('dd')):
                    label = dt.get_text(strip=True)
                    paras = dd.find_all('p')
                    text = '\n'.join(p.get_text(strip=True) for p in paras) if paras else dd.get_text(' ', strip=True)
                    rec[label] = text

            records.append(rec)
    time.sleep(0.2)

# 마무리 및 저장
driver.quit()
df = pd.DataFrame(records)
fixed = ['kind', 'contentId', '질병명', '영문명', '증상_keyword']
others = [c for c in df.columns if c not in fixed]
df = df[fixed + others]
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"✅ 완료: {OUTPUT_FILE} ({len(df)} records)")

