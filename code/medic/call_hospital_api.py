import pandas as pd
import requests
import time
import urllib.parse

# 파일 경로 바꾸기
file_path = "~/temp/add-more/data/hospital_code.csv"

# CSV 불러오기
df = pd.read_csv(file_path, encoding='cp949')

# 컬럼명 변경
df.rename(columns={
    "암호화된요양기호": "hospital_id",
    "요양기관명": "hospital_name",
    "요양종별": "hospital_type",
    "시도명": "province",
    "시군구명": "city",
    "도로명주소": "address",
    "표시과목명": "medical_subjects",
    "개설일자": "establishment_date"
}, inplace=True)

# 제외할 병원 유형
exclude_types = ['한의원', '한방병원', '요양병원', '약국']

# 제외한 데이터프레임 만들기
filtered_df = df[~df['hospital_type'].isin(exclude_types)]

API_KEY = "JEfO7+ak/zek1eGZYyaelPf6YBUj2KndPEaE2r6u3Ps87lmtYxLRliaCYt53+4A5wYIQFuA8kFo8amGf205ekg=="
url = 'http://apis.data.go.kr/B551182/MadmDtlInfoService2.7/getDgsbjtInfo2.7'

# treatment 저장할 리스트
treatment_results = []


for row in filtered_df.itertuples(index=False):
    ykiho = row.hospital_id

    # API 요청 파라미터 설정
    params = {
        'serviceKey': API_KEY,
        'ykiho': ykiho,
        '_type': 'json'
    }

    try:
        res = requests.get(url, params=params)
        data = res.json()

        items = data['response']['body']['items']

        # 'item'이 존재할 때 처리
        if 'item' in items:
            item_data = items['item']
            if isinstance(item_data, dict):  # 단일 항목일 경우
                item_data = [item_data]

            treatment_list = [item['dgsbjtCdNm'] for item in item_data]
            treatments = ", ".join(treatment_list)
        else:
            treatments = ""

    except Exception as e:
        treatments = ""
        print(f"❌ {res.url} 처리 중 오류 발생: {e}")

    treatment_results.append(treatments)
    print(ykiho, treatments)  # 진행 상황 출력


# 진료과목 컬럼 추가
filtered_df["treatment"] = treatment_results
# 저장할 컬럼 선택택
columns_to_save = ["hospital_name", "hospital_type", "province", "city", "address", "treatment"]
# 저장
filtered_df[columns_to_save].to_csv("~/temp/add-more/data/hospital_filtered.csv", index=False, encoding="utf-8-sig")
                                                                                                                                 
