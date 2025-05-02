# step3_clean_and_check.py

import pandas as pd
import re

INPUT_CSV = "asan_disease_data.csv"
OUTPUT_CSV = "asan_disease_filtered.csv"
ERROR_CSV = "asan_disease_format_issues.csv"

# 불러오기
df = pd.read_csv(INPUT_CSV)

# ✅ 질환명 분리 (한글, 영어)
def split_title(title):
    match = re.match(r"^(.*?)(?:\((.*?)\))?$", str(title))
    if match:
        return match.group(1).strip(), match.group(2).strip() if match.group(2) else ""
    return title, ""

df["질환명_한글"], df["질환명_영문"] = zip(*df["질환명"].map(split_title))

# ✅ 필요한 컬럼만 추출
columns_keep = [
    "카테고리ID" if "카테고리ID" in df.columns else None,
    "질환명_한글", "질환명_영문", "증상_Keyword", "정의",
    "원인", "증상", "진단", "치료", "경과", "출처_URL"
]
columns_keep = [col for col in columns_keep if col in df.columns]  # None 제거
filtered_df = df[columns_keep]

# ✅ 이상 항목 탐지 (예: 질환명_영문 없음, keyword가 너무 짧거나 없음 등)
def detect_issue(row):
    issues = []
    if not row["질환명_영문"]:
        issues.append("영문명 없음")
    if pd.isna(row["증상_Keyword"]) or len(str(row["증상_Keyword"]).split(",")) < 2:
        issues.append("증상_Keyword 형식 이상")
    if not str(row["출처_URL"]).startswith("https://www.amc.seoul.kr"):
        issues.append("URL 이상")
    return ", ".join(issues)

filtered_df["형태_이상_여부"] = filtered_df.apply(detect_issue, axis=1)

# ✅ 저장
filtered_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

# 이상 행만 별도 저장
filtered_df[filtered_df["형태_이상_여부"] != ""].to_csv(ERROR_CSV, index=False, encoding="utf-8-sig")

print("✅ 주요 컬럼 정리 및 이상값 검사 완료")