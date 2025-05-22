
"""
증상 중심 고순도 의학 토큰화 모듈
- 증상(sym) 컬럼만 사용
- 복합명사 + 의학사전 최대 활용
- 명사 중심 토큰화
- 부정문 처리 제외 (API에서 처리됨)
"""

import os
import re
import pandas as pd
import json
import numpy as np
from konlpy.tag import Okt
import logging
from datetime import datetime
from typing import List, Dict, Set, Tuple
from collections import Counter

# 파일 경로 설정
DISEASE_CSV_PATH = "/Users/jacob/Desktop/token/disease_data.csv"
MED_TERMS_CSV_PATH = "/Users/jacob/Desktop/token/medical_terms_cleaned.csv"
OUTPUT_CSV_PATH = "/Users/jacob/Desktop/token/processed_disease_data_v2.csv"

# 로그 디렉토리 생성
LOG_DIR = "/Users/jacob/Desktop/token/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 로깅 설정
log_filename = os.path.join(LOG_DIR, f"tokenization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 불용어 목록 (기존 재사용 + 증상 특화)
STOPWORDS = [
    "가능성", "가장", "가지", "감소", "감염", "갑자기", "개월", "거나", "거의",
    "검사", "결과", "결정", "결핍", "경과", "경향", "경험", "고려", "공간",
    "공급", "과도", "과정", "관련", "관절", "관찰", "교정", "구분", "구조",
    "국소", "급성", "기간", "기는", "기능", "기면", "나이",
    "내부", "노출", "능력", "다른", "다시", "다음", "달리", "대개", "대부분", "대표",
    "대한", "대해", "도움", "동반", "동안", "따라서", "때문", "또한", "마비",
    "만성", "매우", "면역", "모두", "모든", "모양", "목적", "문제", "물질",
    "반복", "반응", "발견", "발달", "발병", "발생", "발성",
    "방법", "방사선", "범위", "변이", "변형", "변화", "보고", "보이", "보존", "보통",
    "복용", "부분", "부위", "부족", "부종", "분류",
    "비교", "비정상", "빈도", "사람", "사망", "사용", "사이", "삽입", "상승",
    "상태", "상황", "색소", "서서히", "선택", "성인", "성장",
    "소견", "소실", "소아", "손상", "수도", "수면", "수술", "수의", "수축",
    "시간", "시기", "시도", "시술", "시작", "시행", "신장", "신체",
    "아래", "악화", "안정", "압박", "약물", "약제", "양상", "양성", "여러",
    "여부", "여성", "역할", "연령", "염색", "염증", "영향", "예방", "완화", "외부",
    "요법", "용어", "우리", "우리나라", "우선", "운동", "움직임", "원인", "위축", "위치",
    "위해", "위험", "유발", "유지", "의미", "의사", "의식", "의심", "이내", "이로","구급차",
    "이루", "이상", "이식", "이외", "이용", "이전", "이하", "이후", "일반", "일부","항상","타고",
    "일시", "임상", "자극", "자연", "자주", "자체", "작용", "장기", "장애", "재발","오히려",
    "저하", "전신", "전체", "절개", "절제", "점차", "정도", "정상","특이","겉보기","반드시",
    "정의", "제거", "제한", "조기", "조절", "존재", "종류","촬영","보아","진전","반면","저절로",
    "주로", "주변", "주사", "주요", "주위", "중이", "증가", "증상", "증식","전달",
    "지속", "지연", "직접", "진단", "진행", "질병", "질환", "차이", "차지", "처음",
    "체내", "초기", "초래", "최근", "출생", "치료", "치료법", "침범","외국","조사",
    "크게", "크기", "통증", "통해", "투여", "특징", "포함", "표면", "피로","인구","상대",
    "필요", "하나", "항생제", "해당", "행동", "현상", "현재","정기","주관","바로",
    "형성", "형태", "호르몬", "호소", "호전", "확장", "환자", "활동", "회복","수년",
    "효과","하와"
    # 일반 한국어 불용어
    "있다", "없다", "되다", "하다", "이다", "같다", "때문", "따라서",
    "그러나", "하지만", "또한", "등", "및", "에서", "으로", "이나","때로는",
    "그리고", "또는", "경우", "통해", "위해", "대해", "이런", "그런","가운데","한편"
    # 한국어 조사/접속사/어미
    "의", "에", "을", "를", "은", "는", "이", "가", "와", "과", "로", "으로", 
    "에서", "부터", "까지", "처럼", "만큼", "보다", "같이", "같은", "이나", 
    "거나", "더", "좀", "잘", "못",
    # 증상 관련 일반 용어 (의미 희석 방지)
    "증상", "질병", "질환", "환자", "치료", "상태", "경우", "정도", "부분"
]

# 복합명사 목록 (기존 재사용)
COMPOUND_NOUNS = [
    "급성 위염", "만성 위염", "급성 폐렴", "만성 폐렴", "급성 기관지염", "만성 기관지염",
    "급성 충수염", "만성 충수염", "급성 신우신염", "만성 신우신염", "급성 간염", "만성 간염",
    "급성 췌장염", "만성 췌장염", "급성 담낭염", "담낭 결석", "담석증", "신장 결석",
    "요관 결석", "방광 결석", "요로 결석", "알레르기성 비염", "알레르기 비염",
    "알레르기 결막염", "알레르기 피부염", "아토피 피부염", "접촉성 피부염",
    "당뇨병", "제1형 당뇨병", "제2형 당뇨병", "인슐린 의존성 당뇨병", "인슐린 비의존성 당뇨병",
    "당뇨병성 케톤산증", "당뇨병성 신증", "당뇨병성 망막병증", "당뇨병성 신경병증",
    "고혈압", "본태성 고혈압", "이차성 고혈압", "폐 고혈압", "문맥 고혈압",
    "관상동맥 질환", "관상동맥 경화증", "관상동맥 협착증", "심근 경색", "심근 허혈",
    "심장 판막 질환", "심방 세동", "심실 세동", "심부전", "울혈성 심부전",
    "뇌졸중", "허혈성 뇌졸중", "출혈성 뇌졸중", "일과성 허혈 발작", "뇌출혈",
    "지주막하 출혈", "경막하 출혈", "경막외 출혈", "뇌내 출혈", "뇌혈관 질환",
    "위식도 역류 질환", "위식도 역류", "위궤양", "십이지장 궤양", "소화성 궤양",
    "위장관 출혈", "장 폐색", "대장 용종", "대장 폴립", "과민성 장 증후군",
    "크론병", "궤양성 대장염", "염증성 장 질환", "만성 설사", "변비",
    "갑상선 기능 항진증", "갑상선 기능 저하증", "갑상선염", "하시모토 갑상선염",
    "갑상선 결절", "갑상선 암", "부신 기능 부전", "쿠싱 증후군", "갈색세포종",
    "류마티스 관절염", "골관절염", "강직성 척추염", "통풍", "섬유근통",
    "건선", "습진", "두드러기", "편두통", "긴장성 두통", "군발성 두통",
    "삼차신경통", "안면 신경 마비", "대상 포진", "대상 포진 후 신경통",
    "천식", "만성 폐쇄성 폐질환", "만성 기관지염", "폐기종", "폐렴",
    "폐색전증", "폐결핵", "간질성 폐질환", "폐섬유증", "기흉",
    "간염", "간경화", "지방간", "알코올성 간질환", "비알코올성 지방간염",
    "담석증", "담낭염", "담낭 폴립", "췌장염", "췌장암",
    "신장염", "신우신염", "신부전", "만성 신장병", "신증후군",
    "요로 감염", "방광염", "전립선 비대증", "전립선염", "전립선암",
    "유방 섬유선종", "유방염", "유방암", "자궁 근종", "자궁내막증",
    "자궁경부염", "자궁경부암", "난소낭종", "난소암", "질염",
    "우울증", "공황 장애", "불안 장애", "강박 장애", "외상후 스트레스 장애",
    "양극성 장애", "정신분열증", "치매", "알츠하이머병", "파킨슨병","부부관계",
    "다발성 경화증", "근위축성 측삭 경화증", "헌팅턴병", "길랭-바레 증후군",
    "중증 근무력증", "간질", "뇌전증", "편두통", "현훈증", "메니에르병","발음장애","의식장애",
    "백내장", "녹내장", "황반변성", "망막박리", "결막염","한쪽마비","감각저하",
    "부비동염", "편도염", "인두염", "후두염", "중이염","근위지골",
    "감기", "독감", "폐렴", "기관지염", "후두염","호흡곤란","중수골","중족골의 단축",
    "충수염", "대장염", "게실염", "췌장염", "담낭염","작열감","근육마비","신경 마비",
    "HIV 감염", "에이즈", "결핵", "말라리아", "뎅기열","시야장애","근위 지골의 단축",
    "골절", "탈구", "염좌", "타박상", "열상","잇몸출혈","안검하수","기억장애","정신 지체",
    "화상", "동상", "일사병", "열사병", "감전","피하출혈","전향성 기억상실","후향성 기억상실","중위 지골의 단축",
    "COVID-19", "코로나19", "중증 급성 호흡기 증후군", "중동 호흡기 증후군", "에볼라 바이러스병","원위 지골의 단축",
    "급성 심근경색", "급성 췌장염", "급성 담낭염", "급성 신부전", "급성 호흡 곤란 증후군","권태감","소화관 출혈",
]

class DataLoader:
    """데이터 로딩 및 검증 클래스"""
    
    @staticmethod
    def load_disease_data(file_path: str) -> pd.DataFrame:
        """질병 데이터 CSV 로드"""
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"질병 데이터 로드 성공: {len(df)}개 레코드")
            
            # 필요한 컬럼 확인
            required_cols = ['disnm_ko', 'disnm_en', 'dep', 'def', 'sym']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.error(f"필수 컬럼 누락: {missing_cols}")
                return None
                
            return df
        except Exception as e:
            logger.error(f"질병 데이터 로드 실패: {e}")
            return None
    
    @staticmethod
    def load_medical_terms(file_path: str) -> Set[str]:
        """의학용어 사전 로드"""
        medical_terms = set()
        
        try:
            if not os.path.exists(file_path):
                logger.warning(f"의학용어 파일 없음: {file_path}")
                return medical_terms
                
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"의학용어 사전 로드: {len(df)}개 행")
            
            # medterm 컬럼에서 한글 의학용어 추출
            if 'medterm' in df.columns:
                for _, row in df.iterrows():
                    if pd.notna(row['medterm']):
                        term = str(row['medterm']).strip()
                        if term and len(term) > 1:
                            medical_terms.add(term)
                            # 복합명사도 추가
                            if ' ' in term or '-' in term:
                                COMPOUND_NOUNS.append(term.replace('-', ' '))
            
            logger.info(f"의학용어 {len(medical_terms)}개 로드 완료")
            return medical_terms
            
        except Exception as e:
            logger.error(f"의학용어 로드 실패: {e}")
            return medical_terms

class TextPreprocessor:
    """텍스트 전처리 클래스"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """텍스트 전처리"""
        if not isinstance(text, str):
            return ""
        
        # 괄호 내용 제거 (한글 포함된 괄호는 보존)
        text = re.sub(r'\([^가-힣]*\)', '', text)
        
        # 숫자 + 단위 제거
        text = re.sub(r'\d+(?:mg|ml|회|번|시간|일|주|개월|년|도|℃)', '', text, flags=re.IGNORECASE)
        
        # 순수 숫자 제거
        text = re.sub(r'\b\d+\b', '', text)
        
        # 특수문자 제거 (한글, 영문, 공백, 하이픈만 보존)
        text = re.sub(r'[^\w가-힣ㄱ-ㅎㅏ-ㅣ\s\-]', ' ', text)
        
        # 중복 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

class SymptomTokenizer:
    """증상 중심 토큰화 클래스"""
    
    def __init__(self, medical_terms: Set[str]):
        """토크나이저 초기화"""
        self.tokenizer = Okt()
        self.medical_terms = medical_terms
        self.compound_nouns = set(COMPOUND_NOUNS)
        self.stopwords = set(STOPWORDS)
        
        # 의학용어를 복합명사에도 추가
        for term in medical_terms:
            if ' ' in term or len(term) >= 3:
                self.compound_nouns.add(term)
        
        logger.info(f"토크나이저 초기화: 의학용어 {len(self.medical_terms)}개, 복합명사 {len(self.compound_nouns)}개")
    
    def extract_compound_terms(self, text: str) -> List[str]:
        """복합 의학용어 추출"""
        found_compounds = []
        
        # 기존 복합명사 매칭 (긴 것부터 우선)
        sorted_compounds = sorted(self.compound_nouns, key=len, reverse=True)
        for compound in sorted_compounds:
            if compound in text and compound not in found_compounds:
                found_compounds.append(compound)
        
        # 패턴 기반 복합용어 추출
        patterns = [
            r'([\w가-힣]+\s+(?:증후군|질환|장애|증상|질병|염증|감염|결석|궤양|암))',
            r'((?:급성|만성|알레르기성|염증성|감염성)\s+[\w가-힣]+)',
            r'([\w가-힣]+성\s+[\w가-힣]+)',
            r'(제\d+형\s+[\w가-힣]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match not in found_compounds and len(match) > 3:
                    found_compounds.append(match)
        
        return found_compounds
    
    def extract_medical_terms(self, text: str) -> List[str]:
        """의학용어 추출"""
        found_terms = []
        
        # 의학용어 사전에서 매칭
        for term in self.medical_terms:
            if term in text and term not in found_terms:
                found_terms.append(term)
        
        # 의학 패턴 추출 (증상 관련)
        symptom_patterns = [
            r'([\w가-힣]{2,}통)',      # 두통, 복통, 요통 등
            r'([\w가-힣]{2,}열)',      # 발열, 미열 등
            r'([\w가-힣]{2,}증)',      # 어지럼증, 울혈증 등
            r'([\w가-힣]{2,}염)',      # 위염, 간염, 폐렴 등
            r'([\w가-힣]{2,}병)',      # 당뇨병, 고혈압 등
        ]
        
        for pattern in symptom_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match not in found_terms and len(match) >= 2:
                    found_terms.append(match)
        
        return found_terms
    
    def is_valid_token(self, token: str) -> bool:
        """토큰 유효성 검사"""
        # 의학용어나 복합명사면 유효
        if token in self.medical_terms or token in self.compound_nouns:
            return True
        
        # 불용어면 무효
        if token in self.stopwords:
            return False
        
        # 길이 체크
        if len(token) < 2:
            return False
        
        # 한글 명사인지 확인
        if not re.match(r'^[가-힣]+$', token):
            return False
        
        return True
    
    def tokenize_symptoms(self, symptoms_text: str) -> List[str]:
        """증상 텍스트 토큰화"""
        if not symptoms_text or not isinstance(symptoms_text, str):
            return []
        
        # 전처리
        cleaned_text = TextPreprocessor.clean_text(symptoms_text)
        if not cleaned_text:
            return []
        
        # 1. 복합용어 추출 (우선순위 높음)
        compound_terms = self.extract_compound_terms(cleaned_text)
        
        # 2. 의학용어 추출
        medical_terms = self.extract_medical_terms(cleaned_text)
        
        # 3. 추출된 용어들을 임시 제거하여 중복 방지
        temp_text = cleaned_text
        for term in compound_terms + medical_terms:
            temp_text = re.sub(rf'\b{re.escape(term)}\b', ' ', temp_text)
        temp_text = re.sub(r'\s+', ' ', temp_text).strip()
        
        # 4. 나머지 텍스트에서 명사 추출
        nouns = []
        if temp_text:
            nouns = self.tokenizer.nouns(temp_text)
        
        # 5. 모든 토큰 합치기
        all_tokens = compound_terms + medical_terms + nouns
        
        # 6. 필터링 및 중복 제거
        final_tokens = []
        for token in all_tokens:
            if token and token not in final_tokens and self.is_valid_token(token):
                final_tokens.append(token)
        
        return final_tokens

class QualityManager:
    """토큰화 품질 관리 클래스"""
    
    @staticmethod
    def calculate_statistics(results: List[Dict]) -> Dict:
        """토큰화 결과 통계 계산"""
        if not results:
            return {}
        
        token_counts = []
        total_tokens = []
        
        for result in results:
            if 'tokens' in result and isinstance(result['tokens'], list):
                token_count = len(result['tokens'])
                token_counts.append(token_count)
                total_tokens.extend(result['tokens'])
        
        unique_tokens = set(total_tokens)
        token_freq = Counter(total_tokens)
        
        stats = {
            'total_records': len(results),
            'avg_tokens_per_record': np.mean(token_counts) if token_counts else 0,
            'max_tokens': max(token_counts) if token_counts else 0,
            'min_tokens': min(token_counts) if token_counts else 0,
            'total_unique_tokens': len(unique_tokens),
            'total_tokens': len(total_tokens),
            'top_10_tokens': token_freq.most_common(10)
        }
        
        return stats
    
    @staticmethod
    def generate_report(stats: Dict, output_path: str):
        """품질 리포트 생성"""
        report = f"""
=== 토큰화 품질 리포트 ===
생성시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 기본 통계:
- 처리된 레코드 수: {stats['total_records']}개
- 레코드당 평균 토큰 수: {stats['avg_tokens_per_record']:.2f}개
- 최대 토큰 수: {stats['max_tokens']}개
- 최소 토큰 수: {stats['min_tokens']}개
- 총 고유 토큰 수: {stats['total_unique_tokens']}개
- 총 토큰 수: {stats['total_tokens']}개

🔝 빈출 토큰 Top 10:
"""
        for i, (token, count) in enumerate(stats['top_10_tokens'], 1):
            report += f"{i:2d}. {token} ({count}회)\n"
        
        report_path = output_path.replace('.csv', '_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"품질 리포트 저장: {report_path}")
        print(report)

def main():
    """메인 실행 함수"""
    logger.info("=== 증상 중심 토큰화 시작 ===")
    
    # 1. 데이터 로드
    logger.info("1. 데이터 로딩...")
    df = DataLoader.load_disease_data(DISEASE_CSV_PATH)
    if df is None:
        logger.error("질병 데이터 로드 실패")
        return
    
    medical_terms = DataLoader.load_medical_terms(MED_TERMS_CSV_PATH)
    
    # 2. 토크나이저 초기화
    logger.info("2. 토크나이저 초기화...")
    tokenizer = SymptomTokenizer(medical_terms)
    
    # 3. 증상이 있는 레코드만 필터링
    symptom_df = df[df['sym'].notna() & (df['sym'].str.strip() != '')].copy()
    logger.info(f"증상 정보 있는 레코드: {len(symptom_df)}개 (전체 {len(df)}개 중 {len(symptom_df)/len(df)*100:.1f}%)")
    
    # 4. 토큰화 처리
    logger.info("3. 토큰화 처리 시작...")
    results = []
    
    for idx, row in symptom_df.iterrows():
        try:
            # 증상 텍스트 토큰화
            symptoms = str(row['sym']) if pd.notna(row['sym']) else ""
            tokens = tokenizer.tokenize_symptoms(symptoms)
            
            # 결과 레코드 생성
            result = {
                'id': idx,
                'disnm_ko': str(row.get('disnm_ko', '')),
                'disnm_en': str(row.get('disnm_en', '')),
                'dep': str(row.get('dep', '')),
                'def': str(row.get('def', '')),
                'symptoms': symptoms,
                'therapy': str(row.get('therapy', '')),
                'tokens': tokens,
                'def_k': ' '.join(tokens),  # TF-IDF용 문자열
                'tokens_json': json.dumps(tokens, ensure_ascii=False)  # CSV 저장용
            }
            results.append(result)
            
            # 진행상황 로깅
            if (len(results)) % 100 == 0:
                logger.info(f"처리 진행: {len(results)}/{len(symptom_df)} ({len(results)/len(symptom_df)*100:.1f}%)")
                
        except Exception as e:
            logger.error(f"레코드 {idx} 처리 오류: {e}")
    
    # 5. 결과 저장
    logger.info("4. 결과 저장...")
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
    logger.info(f"✅ 토큰화 결과 저장: {OUTPUT_CSV_PATH}")
    logger.info(f"✅ 처리 완료: {len(results)}개 레코드")
    
    # 6. 품질 리포트 생성
    logger.info("5. 품질 분석...")
    stats = QualityManager.calculate_statistics(results)
    QualityManager.generate_report(stats, OUTPUT_CSV_PATH)
    
    logger.info("=== 토큰화 완료 ===")

if __name__ == "__main__":
    main()