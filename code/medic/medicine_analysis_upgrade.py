import pandas as pd
import os
import json
import urllib3
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# DB 연결 함수
def get_engine():
    db_url = os.getenv("DATABASE_URL")
    return create_engine(db_url)

# 형태소 분석 API 
accessKey = os.getenv("ETRI_API_KEY")
openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"

# ETRI API 호출
def call_etri_api(text):
    analysisCode = "morp"
    requestJson = {
        "access_key": accessKey,
        "argument": {
            "text": text,
            "analysis_code": analysisCode
        }
    }

    http = urllib3.PoolManager()
    response = http.request(
        "POST",
        openApiURL,
        headers={
            "Content-Type": "application/json; charset=UTF-8",
            "Authorization": accessKey
        },
        body=json.dumps(requestJson)
    )
    if response.status != 200:
        return None

    return json.loads(response.data.decode("utf-8"))

# 형태소 분석 결과에서 NNG만 추출
def extract_nng(text):
    if not text:
        print("⚠️ 빈 문자열 입력")
        return ""
    
    json_data = call_etri_api(text)
    if not json_data:
        print("⚠️ API 호출 실패 또는 응답 없음")
        return ""
    
    try:
        result = set()
        sentences = json_data.get('return_object', {}).get('sentence', [])
        if not sentences:
            print("⚠️ 문장 정보 없음:", json_data)
        
        for sentence in sentences:
            morp_eval = sentence.get('morp_eval', [])
            for morp in morp_eval:
                # 'result' 예: "약/NNG+은/JX"
                result_str = morp.get('result', '')
                # '+'로 분리해서 각각 "단어/품사" 형태 분해
                for part in result_str.split('+'):
                    if '/' in part:
                        word, pos = part.split('/', 1)
                        if pos == 'NNG':
                            if word == '약':
                                continue  # "약"은 제외
                            elif word == '육체피':
                                result.add('육체피로')  # "육체피"는 "육체피로"로 대체
                            else:
                                result.add(word)
        
        nng_result = ', '.join(sorted(result))
        return nng_result
    except Exception as e:
        print(f"🔴 추출 중 오류 발생: {e}")
        return ""
    
# DB에서 데이터 불러오기 및 ph_effect_c 업데이트
def update_ph_effect_c():
    engine = get_engine()
    
    with engine.begin() as conn:  # 트랜잭션 자동 커밋
        df = pd.read_sql("SELECT ph_nm, ph_effect FROM testmed", conn)
        df['ph_effect'] = df['ph_effect'].fillna("")
        
        print("형태소 분석 및 DB 업데이트 시작...")

        for idx, row in df.iterrows():
            nng_result = extract_nng(row['ph_effect'])
            
            print(f"[{row['ph_nm']}] NNG 추출 결과: {nng_result}")  # 🔍 확인용 출력

            update_query = text("""
                UPDATE testmed 
                SET ph_effect_c = :nng_result 
                WHERE ph_nm = :ph_nm
            """)
            conn.execute(update_query, {"nng_result": nng_result, "ph_nm": row['ph_nm']})
        
        print("형태소 분석 및 ph_effect_c 컬럼 업데이트 완료.")
        
        
# 실행
if __name__ == "__main__":
    update_ph_effect_c()
