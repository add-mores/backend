import requests
import pandas as pd
import time

API_KEY = "JEfO7+ak/zek1eGZYyaelPf6YBUj2KndPEaE2r6u3Ps87lmtYxLRliaCYt53+4A5wYIQFuA8kFo8amGf205ekg=="
url = 'http://apis.data.go.kr/1471000/DrbEasyDrugInfoService/getDrbEasyDrugList'

all_items = []
page = 1

while True:
    print(f"{page} 페이지 요청")
    params = {
        'serviceKey': API_KEY,
        'pageNo': page,
        'numOfRows': 100,
        'type': 'json'
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        print("❌ 요청 실패:", response.status_code)
        break

    data = response.json()

    try:
        items = data['body']['items']
    except KeyError:
        print("✅ 모든 페이지 수집 완료.")
        break

    all_items.extend(items)
    page += 1
    time.sleep(0.3)

# DataFrame 저장 위치 정해지면 바꾸기! 
df = pd.DataFrame(all_items)
df.to_csv('medicine_info_all.csv', index=False, encoding='utf-8-sig')
print(f"총 {len(df)}건 저장 완료")
