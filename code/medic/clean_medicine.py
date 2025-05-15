import pandas as pd
import re

# 기존 CSV 파일 불러오기 (파일 경로 바꾸기)
df = pd.read_csv("~/temp/add-more/data/medicine_info_all.csv")

# 원하는 컬럼만 선택
columns_to_keep = [
    "entpName",               # 업체명
    "itemName",               # 제품명
    "efcyQesitm",             # 효능
    "useMethodQesitm",        # 사용법
    "atpnWarnQesitm",         # 주의사항 경고
    "atpnQesitm",             # 주의사항
    "seQesitm",               # 부작용
    "depositMethodQesitm"     # 보관법
]

# 해당 컬럼들만 추출 (없는 컬럼이 있다면 오류 방지를 위해 존재하는 것만 필터링)
filtered_df = df[[col for col in columns_to_keep if col in df.columns]].copy()

EXCEPTION_PATTERNS = [
    "향", "순", "쿨", "1회용", "쿨하이", "순-1회용", "쿨-1회용", "라이트"
]

# 하이픈 제거된 형태의 매핑 생성
EXCEPTION_RESTORE_MAP = {
    pat.replace("-", ""): pat for pat in EXCEPTION_PATTERNS if "-" in pat
}

def clean_item_name(name):
    # 1. '수출명' 포함 시 그 이후 제거
    name = re.split(r"수출명", name)[0]

    # 2. 예외 괄호 통째로 보존 (괄호 제거, 하이픈 제거 형태로 저장)
    for pat in EXCEPTION_PATTERNS:
        safe_pat = pat.replace("-", "")
        name = name.replace(f"({pat})", f"__KEEP__{safe_pat}__")

    # 3. 중첩 괄호 제거
    while re.search(r"[\(\[{<][^()\[\]{}<>]*[\)\]}>]", name):
        name = re.sub(r"[\(\[{<][^()\[\]{}<>]*[\)\]}>]", "", name)

    # 4. 숫자. → 쉼표로 (소수점은 제외)
    name = re.sub(r"(^|\D)(\d+)\.\s*(?=[가-힣a-zA-Z])", r"\1,", name)

    # 5. %, ., , 외 특수기호 제거
    name = re.sub(r"[^\w\s.,%]", "", name)

    # 6. 쉼표 정리
    name = re.sub(r",+", ",", name)

    # 7. 앞뒤 쉼표, 공백 제거
    name = name.strip(", ").strip()

    # 8. 복원: 괄호 포함 형태로 되돌리기
    for safe_pat in EXCEPTION_RESTORE_MAP:
        original = EXCEPTION_RESTORE_MAP[safe_pat]
        name = name.replace(f"__KEEP__{safe_pat}__", f"({original})")
        
        # 나머지 예외들도 복원
    for pat in EXCEPTION_PATTERNS:
        if "-" not in pat:  # 하이픈 없는 건 그냥 복원
            name = name.replace(f"__KEEP__{pat}__", f"({pat})")

    return name

# 클리닝 적용
filtered_df.loc[:, "itemName_clean"] = filtered_df["itemName"].apply(clean_item_name)

# 새 CSV 파일로 저장 (파일 경로 바꾸기)
filtered_df.to_csv("~/temp/add-more/data/medicine_info_cleaned.csv", index=False, encoding="utf-8-sig")

print("cleaning done")
