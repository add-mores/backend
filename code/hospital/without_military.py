import pandas as pd
import re

# 1) 실패한 주소 불러오기
df = pd.read_csv("missing_hospitals_failed_retry.csv")

# 2) '교도소' 및 군대 관련 키워드 정의
keywords = [
    "교도소", "보호관찰소", "교정",          # 교정시설
    "군부대", "부대", "사단", "여단",       # 군부대
    "육군", "해군", "공군", "해병대",       # 군종
    "군사", "훈련소", "병영"                # 기타 군사시설
]

# 3) 해당 키워드를 포함하는 행 식별
pattern = "|".join(keywords)
mask_known = df["address"].str.contains(pattern)

# 4) 교도소/군대 관련을 제외한 나머지
others = df[~mask_known].copy()

# 5) 결과 확인
print("✅ 교정시설 및 군대 관련 주소 개수:", mask_known.sum())
print("📌 기타 실패 주소 개수:", len(others))
print(others[["address", "cleaning"]].to_string(index=False))

# 6) 기타 실패 주소만 별도 CSV로 저장
others.to_csv("missing_hospitals_failed_other_categories.csv", index=False, encoding="utf-8-sig")

