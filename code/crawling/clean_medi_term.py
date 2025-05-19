import pandas as pd
import re
import os

# 입력 파일 및 출력 파일 경로 설정
input_csv = 'medical_terms.csv'  # 기존 CSV 파일 경로
output_csv = 'medical_terms_cleaned.csv'  # 새로 저장할 CSV 파일 경로

# CSV 파일 읽기 (UTF-8 인코딩으로)
df = pd.read_csv(input_csv, encoding='utf-8')

# 새로운 칼럼 추가
df['meditem_cleaned'] = ''
df['medterm_eng'] = ''

# 각 행에 대해 처리
for idx, row in df.iterrows():
    meditem = row['meditem']
    
    # 영문명 추출 (마지막 괄호 안의 텍스트)
    # 패턴: (영문명) 또는 (영문명(약어))
    eng_match = re.search(r'\(([^()]+(?:\([^()]+\))?)\)$', meditem)
    
    if eng_match:
        eng_term_raw = eng_match.group(1)
        
        # 영문명(약어) 형태인지 확인
        nested_match = re.search(r'(.+)\((.+)\)', eng_term_raw)
        
        if nested_match:
            # 영문명(약어) 형태일 경우 "영문명,약어" 형태로 저장
            main_term = nested_match.group(1).strip()
            abbr = nested_match.group(2).strip()
            eng_term = f"{main_term},{abbr}"
        else:
            # 일반 영문명일 경우
            eng_term = eng_term_raw
            
        df.at[idx, 'medterm_eng'] = eng_term
    
    # 한글명 클리닝
    # 1. 영문명 괄호 제거: 마지막에 있는 영문명 괄호와 그 내용 제거
    term_without_eng = re.sub(r'\([^()]+(?:\([^()]+\))?\)$', '', meditem).strip()
    
    # 2. 모든 나머지 괄호와 그 내용 제거 (술, 법, 증 등) + '-' 제거
    cleaned_term = re.sub(r'[-\s]*\([^()]+\)', '', term_without_eng).strip()
    
    # 3. 혹시 남아있는 하이픈 제거
    cleaned_term = cleaned_term.replace('-', '').strip()
    
    df.at[idx, 'meditem_cleaned'] = cleaned_term

# 칼럼 순서 재정렬: 원래칼럼(meditem), 수정한글칼럼(meditem_cleaned), 영문칼럼(medterm_eng), 나머지
original_cols = df.columns.tolist()
original_cols.remove('meditem_cleaned')
original_cols.remove('medterm_eng')

# 원하는 순서대로 칼럼 재배열
new_cols = ['meditem', 'meditem_cleaned', 'medterm_eng']
for col in original_cols:
    if col != 'meditem':  # meditem은 이미 추가했으므로 제외
        new_cols.append(col)

df = df[new_cols]

# 결과 확인 (처음 5개 행)
print(df.head())

# Excel에서도 읽을 수 있게 UTF-8 BOM 인코딩으로 저장
df.to_csv(output_csv, encoding='utf-8-sig', index=False)

print(f"전처리가 완료되었습니다. 결과는 '{output_csv}' 파일에 저장되었습니다.")
