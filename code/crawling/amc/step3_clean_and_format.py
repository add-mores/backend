# step3_clean_and_format.py

import pandas as pd
import re

# 파일 경로 (step2 결과물)
input_path = "asan_disease_data.csv"
output_path = "asan_disease_final_cleaned.csv"

# 데이터 불러오기
df = pd.read_csv(input_path)

# ✅ 1. .1 컬럼 제거 (중복 컬럼 제거)
df = df.loc[:, ~df.columns.str.endswith(".1")]

# ✅ 2. 질환명 분리 함수 정의
def split_title(title):
    match = re.match(r"^(.*?)(?:\((.*?)\))?$", str(title))
    if match:
        return match.group(1).strip(), match.group(2).strip() if match.group(2) else ""
    return title, ""

# ✅ 3. 질환명 한글/영문 컬럼 생성 (이미 있으면 건너뜀)
if "질환명" in df.columns and "질환명_한글" not in df.columns:
    df["질환명_한글"], df["질환명_영문"] = zip(*df["질환명"].map(split_title))

# ✅ 4. URL 제거
url_cols = [col for col in df.columns if "URL" in col]
df = df.drop(columns=url_cols, errors="ignore")

# ✅ 5. 컬럼 순서 정리
cols = df.columns.tolist()
core_cols = ["contentId", "질환명_한글", "질환명_영문"]
ordered_cols = core_cols + [col for col in cols if col not in core_cols]
df = df[ordered_cols]

# ✅ 6. 저장
df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"✅ 최종 정제 완료: {output_path}")
print(df.head(10))
