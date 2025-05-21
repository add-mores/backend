import os
import re
import pandas as pd
import json
import numpy as np
from konlpy.tag import Okt
import logging
from datetime import datetime

# 파일 경로
DISEASE_CSV_PATH = "/Users/jacob/Desktop/token/disease_data.csv"  
MED_TERMS_CSV_PATH = "/Users/jacob/Desktop/token/medical_terms_cleaned.csv" 
OUTPUT_CSV_PATH = "/Users/jacob/Desktop/token/processed_disease_data_v2.csv" 

# 로깅 설정
log_filename = f"tokenizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler(log_filename, encoding='utf-8'),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger(__name__)

# 수정된 불용어 목록
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
    "위해", "위험", "유발", "유지", "의미", "의사", "의식", "의심", "이내", "이로",
    "이루", "이상", "이식", "이외", "이용", "이전", "이하", "이후", "일반", "일부",
    "일시", "임상", "자극", "자연", "자주", "자체", "작용", "장기", "장애", "재발",
    "저하", "전신", "전체", "절개", "절제", "점차", "정도", "정상",
    "정의", "제거", "제한", "조기", "조절", "존재", "종류",
    "주로", "주변", "주사", "주요", "주위", "중이", "증가", "증상", "증식",
    "지속", "지연", "직접", "진단", "진행", "질병", "질환", "차이", "차지", "처음",
    "체내", "초기", "초래", "최근", "출생", "치료", "치료법", "침범",
    "크게", "크기", "통증", "통해", "투여", "특징", "포함", "표면", "피로",
    "필요", "하나", "항생제", "해당", "행동", "현상", "현재",
    "형성", "형태", "호르몬", "호소", "호전", "확장", "환자", "활동", "회복",
    "효과"
    # 일반 한국어 불용어 추가
    "있다", "없다", "되다", "하다", "이다", "같다", "때문", "따라서",
    "그러나", "하지만", "또한", "등", "및", "에서", "으로", "이나",
    "그리고", "또는", "경우", "통해", "위해", "대해", "이런", "그런",
    # 한국어 조사/접속사/어미 등도 추가
    "의", "에", "을", "를", "은", "는", "이", "가", "와", "과", "로", "으로", "에서", "부터",
    "까지", "처럼", "만큼", "보다", "같이", "같은", "이나", "거나", "더", "좀", "잘", "못"
]

# 복합명사 목록
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
    "양극성 장애", "정신분열증", "치매", "알츠하이머병", "파킨슨병",
    "다발성 경화증", "근위축성 측삭 경화증", "헌팅턴병", "길랭-바레 증후군",
    "중증 근무력증", "간질", "뇌전증", "편두통", "현훈증", "메니에르병",
    "백내장", "녹내장", "황반변성", "망막박리", "결막염",
    "부비동염", "편도염", "인두염", "후두염", "중이염",
    "감기", "독감", "폐렴", "기관지염", "후두염",
    "충수염", "대장염", "게실염", "췌장염", "담낭염",
    "HIV 감염", "에이즈", "결핵", "말라리아", "뎅기열",
    "골절", "탈구", "염좌", "타박상", "열상",
    "화상", "동상", "일사병", "열사병", "감전",
    "COVID-19", "코로나19", "중증 급성 호흡기 증후군", "중동 호흡기 증후군", "에볼라 바이러스병",
    "급성 심근경색", "급성 췌장염", "급성 담낭염", "급성 신부전", "급성 호흡 곤란 증후군"
]

# 뼈 이름 (골로 끝나는 용어) 목록
BONE_TERMS = [
    "두개골", "안면골", "관골", "광대골", "측두골", "전두골", "후두골", "두정골",
    "접형골", "사골", "비골", "상악골", "하악골", "설골", "경추골", "흉추골",
    "요추골", "천골", "미골", "흉골", "늑골", "쇄골", "견갑골", "상완골",
    "요골", "척골", "수근골", "중수골", "지골", "장골", "좌골", "치골",
    "대퇴골", "슬개골", "경골", "비골", "족근골", "중족골", "지골", "척추골"
]

class MedicalTokenizer:
    """
    의학 텍스트 토큰화 모듈 - 명사 위주 고순도화
    - 의학 용어 사전 활용
    - 복합 명사 보존
    - 불용어 처리 강화
    - 특수 패턴 처리 (혈액형, 뼈 이름 등)
    """
    
    def __init__(self):
        """토크나이저 초기화"""
        # Okt 형태소 분석기 초기화
        self.tokenizer = Okt()
        logger.info("Okt 형태소 분석기 초기화 완료")
        
        # 화이트리스트: 의학 용어, 특수 단어 등 보존할 단어 목록 (기본값)
        self.whitelist = set([
            # 기본 의학 용어 및 약어
            "X-ray", "CT", "MRI", "US", "Ultrasound", "PET", "Endoscopy",
            "ECG", "EKG", "EEG", "BP", "HR", "CBC", "BUN", "Antibiotic",
            "COVID-19", "BRCA", "BRCA1", "BRCA2", "XY", "DNA", "ARC",
            
            # 주요 질병 및 증상 관련 용어
            "고혈압", "당뇨병", "천식", "관절염", "우울증", "불안장애",
            "두통", "편두통", "복통", "요통", "발열", "기침", "구토",
            "설사", "변비", "현기증", "어지러움", "발작", "경련"
        ])
        
        # 뼈 이름(골)을 화이트리스트에 추가
        for bone in BONE_TERMS:
            self.whitelist.add(bone)
        
        # 불용어: 하드코딩된 불용어 목록으로 초기화
        self.stopwords = set(STOPWORDS)
        
        # 복합 명사: 하드코딩된 복합 명사 목록으로 초기화
        self.compound_nouns = set(COMPOUND_NOUNS)
        
        # 복합 명사를 화이트리스트에도 추가
        for term in self.compound_nouns:
            self.whitelist.add(term)
        
        # 의학 용어 사전 로드
        self.load_medical_terms()
        
        # 화이트리스트와 불용어 목록 충돌 확인 및 해결
        self.resolve_conflicts()
        
        logger.info(f"화이트리스트 크기: {len(self.whitelist)}")
        logger.info(f"불용어 크기: {len(self.stopwords)}")
        logger.info(f"복합 명사 크기: {len(self.compound_nouns)}")
    
    def resolve_conflicts(self):
        """화이트리스트와 불용어 목록 간의 충돌 해결"""
        conflicts = self.whitelist.intersection(self.stopwords)
        if conflicts:
            logger.warning(f"화이트리스트와 불용어 목록 사이에 {len(conflicts)}개의 충돌이 있습니다.")
            # 충돌 해결: 화이트리스트 우선
            for term in conflicts:
                self.stopwords.remove(term)
            logger.info(f"충돌 해결 후 불용어 크기: {len(self.stopwords)}")
    
    def load_medical_terms(self):
        """의학 용어 CSV 파일에서 의학 용어 로드"""
        try:
            if not os.path.exists(MED_TERMS_CSV_PATH):
                logger.warning(f"의학 용어 파일이 없습니다: {MED_TERMS_CSV_PATH}")
                return
                
            # 의학 용어 CSV 로드
            med_terms_df = pd.read_csv(MED_TERMS_CSV_PATH, encoding='utf-8')
            logger.info(f"의학 용어 CSV 로드 완료: {len(med_terms_df)}개 행")
            
            # 필요한 컬럼 확인
            if 'medterm' in med_terms_df.columns:
                # 한글 의학 용어 추가
                for _, row in med_terms_df.iterrows():
                    if pd.notna(row['medterm']):
                        term = str(row['medterm']).strip()
                        if term and len(term) > 1:
                            self.whitelist.add(term)
                
                # 영문 의학 용어 추가 (있을 경우)
                if 'medterm_eng' in med_terms_df.columns:
                    for _, row in med_terms_df.iterrows():
                        if pd.notna(row['medterm_eng']):
                            eng_term = str(row['medterm_eng']).strip()
                            if eng_term and len(eng_term) > 1:
                                self.whitelist.add(eng_term)
                
                # 동의어 추가 (있을 경우)
                if 'syn' in med_terms_df.columns:
                    for _, row in med_terms_df.iterrows():
                        if pd.notna(row['syn']):
                            synonyms = str(row['syn'])
                            # 동의어가 여러 개일 경우 ';' 또는 ',' 로 구분되어 있을 수 있음
                            for separator in [';', ',']:
                                if separator in synonyms:
                                    for syn in synonyms.split(separator):
                                        syn = syn.strip()
                                        if syn and len(syn) > 1:
                                            self.whitelist.add(syn)
                                            # 공백이 포함된 단어는 복합 명사로도 추가
                                            if ' ' in syn:
                                                self.compound_nouns.add(syn)
                                    break
                            else:  # 구분자가 없는 경우
                                syn = synonyms.strip()
                                if syn and len(syn) > 1:
                                    self.whitelist.add(syn)
                                    if ' ' in syn:
                                        self.compound_nouns.add(syn)
                
                logger.info(f"의학 용어 CSV에서 화이트리스트에 용어 추가 완료")
            else:
                logger.warning(f"의학 용어 CSV에 'medterm' 컬럼이 없습니다")
                
            # 질환명 컬럼이 있는 경우 (medical_terms.csv)
            if '질환명' in med_terms_df.columns:
                for _, row in med_terms_df.iterrows():
                    if pd.notna(row['질환명']):
                        disease_name = str(row['질환명']).strip()
                        if disease_name:
                            self.whitelist.add(disease_name)
                            # 공백 또는 '-'가 포함된 질병명은 복합 명사로 추가
                            if ' ' in disease_name or '-' in disease_name:
                                # '-'를 공백으로 변환하여 추가
                                self.compound_nouns.add(disease_name.replace('-', ' '))
                
                logger.info(f"질환명 컬럼에서 화이트리스트 및 복합 명사에 용어 추가 완료")
            
        except Exception as e:
            logger.error(f"의학 용어 로드 오류: {e}")
    
    def clean_text(self, text):
        """
        텍스트 전처리 (괄호 제거, 숫자 제거, 특수문자 제거 등)
        Args:
            text: 입력 텍스트
        Returns:
            전처리된 텍스트
        """
        if not isinstance(text, str):
            return ""
            
        # 괄호 내용 제거
        text = re.sub(r"\([^가-힣ㄱ-ㅎㅏ-ㅣ]*\)", "", text)
        
        # 숫자 제거 (단위 포함)
        text = re.sub(r"(?i)\b\d+ml?%?\b", "", text)
        text = re.sub(r"\b\d+\b", "", text)
        
        # 특수문자 제거 (한글, 영문, 공백 외)
        text = re.sub(r"[^\w가-힣ㄱ-ㅎㅏ-ㅣ\s]", " ", text)
        
        # 중복 공백 제거
        return re.sub(r"\s+", " ", text).strip()
    
    def extract_special_patterns(self, text):
        """
        특수 패턴 추출 (혈액형, 뼈 이름, 의학 용어 등)
        Args:
            text: 입력 텍스트
        Returns:
            추출된 특수 패턴 목록
        """
        special_tokens = []
        
        # 혈액형 패턴 (A형, B형 등)
        types_regex = re.compile(r"[A-Z][형]|[A-Z][형]")
        for match in types_regex.finditer(text):
            token = match.group(0)
            if token not in special_tokens:
                special_tokens.append(token)
        
        # 역순 혈액형 패턴 (형A, 형B 등)
        reverse_types_regex = re.compile(r"(형|형|형)\s*([A-Z])")
        for match in reverse_types_regex.finditer(text):
            token_type = match.group(1)
            letter = match.group(2)
            token = f"{token_type}{letter}"  # "형A"
            if token not in special_tokens:
                special_tokens.append(token)
        
        # 뼈 이름 패턴 (OO골)
        bone_regex = re.compile(r"([\w가-힣]{1,5}골)\b")
        for match in bone_regex.finditer(text):
            token = match.group(1)
            if token not in special_tokens and len(token) >= 2:
                special_tokens.append(token)
                # 화이트리스트에도 추가
                self.whitelist.add(token)
        
        # 한국어 의학 용어 패턴들
        korean_patterns = [
            r"([\w가-힣]+ 증후군)\b",
            r"([\w가-힣]+ 질환)\b",
            r"([\w가-힣]+ 장애)\b",
            r"([\w가-힣]+ 증상)\b",
            r"([\w가-힣]+ 질병)\b",
            r"([\w가-힣]+ 염증)\b",
            r"([\w가-힣]+ 감염)\b"
        ]
        
        for pattern in korean_patterns:
            for match in re.finditer(pattern, text):
                token = match.group(1)
                if token not in special_tokens and len(token) >= 2:
                    special_tokens.append(token)
                    
        return special_tokens
    
    def extract_medical_terms(self, text):
        """
        의학 용어 추출
        Args:
            text: 입력 텍스트
        Returns:
            추출된 의학 용어 목록
        """
        medical_terms = []
        
        # 복합 명사 추출
        for term in self.compound_nouns:
            if term in text and term not in medical_terms:
                medical_terms.append(term)
        
        # 의학 패턴 추출 (OO증, OO병, OO염 등)
        medical_patterns = [
            r"([\w가-힣]{1,5}증)\b",  # 고혈압증
            r"([\w가-힣]{1,5}병)\b",  # 당뇨병
            r"([\w가-힣]{1,5}염)\b",  # 위염
            r"([\w가-힣]{1,5}통)\b",  # 두통
            r"([\w가-힣]{1,5}열)\b",  # 발열
            r"([\w가-힣]{1,5}애)\b",  # 장애
            r"([\w가-힣]{1,5}상)\b",  # 증상
            r"([\w가-힣]{1,5}환)\b"   # 질환
        ]
        
        for pattern in medical_patterns:
            for match in re.finditer(pattern, text):
                term = match.group(1)
                if term not in medical_terms and len(term) >= 2:
                    medical_terms.append(term)
                    
        return medical_terms
    
    def is_valid_token(self, token):
        """
        유효한 토큰인지 확인
        Args:
            token: 확인할 토큰
        Returns:
            유효 여부
        """
        # 화이트리스트에 있으면 유효
        if token in self.whitelist:
            return True
            
        # 복합 명사에 있으면 유효
        if token in self.compound_nouns:
            return True
            
        # 불용어에 있으면 유효하지 않음
        if token in self.stopwords:
            return False
            
        # 길이가 1이면 유효하지 않음 (의미없는 토큰)
        if len(token) < 2:
            return False
            
        # 한글, 영문, 숫자만 포함된 토큰만 유효
        return bool(re.fullmatch(r"[\w가-힣ㄱ-ㅎㅏ-ㅣ]+", token))
    
    def tokenize(self, text):
        """
        텍스트 토큰화 - 명사 위주 처리
        Args:
            text: 입력 텍스트
        Returns:
            토큰화된 명사 목록
        """
        # 빈 텍스트 처리
        if not text or not isinstance(text, str):
            return []
            
        # 텍스트 전처리
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []
        
        # 1. 화이트리스트 및 복합 명사 먼저 추출 (이들은 보존)
        whitelist_tokens = []
        for token in self.whitelist:
            if token in cleaned_text and token not in whitelist_tokens:
                whitelist_tokens.append(token)
        
        # 2. 특수 패턴 추출 (혈액형, 뼈 이름, 복합 의학 용어 등)
        special_tokens = self.extract_special_patterns(cleaned_text)
        
        # 3. 의학 용어 패턴 추출 (주로 명사 형태)
        medical_tokens = self.extract_medical_terms(cleaned_text)
        
        # 4. 추출된 토큰들을 원본 텍스트에서 제거 (중복 추출 방지)
        temp = cleaned_text
        for w in whitelist_tokens + special_tokens + medical_tokens:
            # 정규식 패턴으로 단어 경계 지정하여 제거
            temp = re.sub(rf"\b{re.escape(w)}\b", " ", temp)
            temp = re.sub(r"\s+", " ", temp).strip()
        
        # 5. 남은 텍스트에서 명사만 추출 (명사 위주 토큰화)
        nouns = []
        if self.tokenizer:
            nouns = self.tokenizer.nouns(temp)  # 명사만 추출
        
        # 6. 모든 토큰 합치기 (이미 명사 위주로 추출됨)
        all_tokens = whitelist_tokens + special_tokens + medical_tokens + nouns
        final_tokens = []
        
        # 7. 중복 제거 및 불용어 필터링
        for token in all_tokens:
            if token and token not in final_tokens and self.is_valid_token(token):
                final_tokens.append(token)
                
        return final_tokens
    
    def process_disease_csv(self):
        """
        질병 데이터 CSV 토큰화 처리
        Returns:
            토큰화된 결과가 저장된 DataFrame
        """
        try:
            # CSV 파일 존재 확인
            if not os.path.exists(DISEASE_CSV_PATH):
                logger.error(f"질병 데이터 CSV 파일이 없습니다: {DISEASE_CSV_PATH}")
                return None
                
            # CSV에서 질병 데이터 로드
            disease_df = pd.read_csv(DISEASE_CSV_PATH, encoding='utf-8')
            logger.info(f"질병 데이터 CSV 로드 완료: {len(disease_df)}개 레코드")
            
            # 필요한 컬럼 확인 및 조정
            required_columns = ["disnm_ko", "disnm_en", "dep", "def", "symptoms", "therapy"]
            for col in required_columns:
                if col not in disease_df.columns:
                    if col == "def" and "definition" in disease_df.columns:
                        disease_df = disease_df.rename(columns={'definition': 'def'})
                    elif col == "symptoms" and "sym" in disease_df.columns:
                        disease_df = disease_df.rename(columns={'sym': 'symptoms'})
                    else:
                        logger.warning(f"필요한 컬럼 '{col}'이 CSV에 없습니다. 빈 컬럼 추가.")
                        disease_df[col] = ""
            
            # 토큰화 및 결과 저장을 위한 리스트
            processed_records = []
            
            # 각 질병 레코드 처리
            for idx, row in disease_df.iterrows():
                try:
                    # 기본 정보 추출
                    disnm_ko = str(row.get("disnm_ko", "")).strip()
                    disnm_en = str(row.get("disnm_en", "")).strip()
                    dep = str(row.get("dep", "")).strip()
                    
                    # 텍스트 필드 추출 및 전처리
                    definition = self.clean_text(str(row.get("def", "")))
                    symptoms = self.clean_text(str(row.get("symptoms", "")))
                    therapy = self.clean_text(str(row.get("therapy", "")))
                    
                    # 각 필드 토큰화 (명사 위주)
                    def_tokens = self.tokenize(definition)
                    symp_tokens = self.tokenize(symptoms)
                    
                    # 치료법은 제외하고 토큰화 (정의와 증상만 사용)
                    # 치료법은 증상-질병 매핑에 노이즈가 될 수 있음
                    # therapy_tokens = self.tokenize(therapy)
                    
                    # 모든 토큰 결합 및 중복 제거 (치료법 토큰 제외)
                    all_tokens = list(set(def_tokens + symp_tokens))
                    
                    # 정의 토큰 문자열 생성 (def_k)
                    def_k = " ".join(def_tokens)
                    
                    # 결과 레코드 생성
                    record = {
                        "disnm_ko": disnm_ko,
                        "disnm_en": disnm_en,
                        "dep": dep,
                        "def": definition,
                        "symptoms": symptoms,
                        "therapy": therapy,
                        "tokens": all_tokens,
                        "def_k": def_k
                    }
                    processed_records.append(record)
                    
                    # 진행 상황 로깅 (100개마다)
                    if (idx + 1) % 100 == 0 or (idx + 1) == len(disease_df):
                        logger.info(f"{idx + 1}/{len(disease_df)} 레코드 처리 완료")
                        
                except Exception as e:
                    logger.error(f"레코드 {idx} ({row.get('disnm_ko', 'unknown')}) 처리 오류: {e}")
                    logger.error(f"오류 세부 정보: {str(e)}")
            
            # DataFrame 생성
            df_processed = pd.DataFrame(processed_records)
            
            # tokens 열을 JSON 문자열로 변환 (CSV 저장 시 필요)
            df_processed['tokens_json'] = df_processed['tokens'].apply(lambda x: json.dumps(x, ensure_ascii=False))
            
            # CSV 파일로 저장
            df_processed.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')
            logger.info(f"✅ {OUTPUT_CSV_PATH}에 {len(df_processed)}개 레코드 저장 완료")
            
            # 토큰 통계 출력
            token_counts = []
            for _, row in df_processed.iterrows():
                if 'tokens' in row and isinstance(row['tokens'], list):
                    token_counts.append(len(row['tokens']))
            
            if token_counts:
                avg_tokens = sum(token_counts) / len(token_counts)
                max_tokens = max(token_counts)
                min_tokens = min(token_counts)
                
                logger.info(f"토큰 통계: 평균 {avg_tokens:.2f}개, 최대 {max_tokens}개, 최소 {min_tokens}개")
            
            return df_processed
            
        except Exception as e:
            logger.error(f"질병 데이터 처리 오류: {e}")
            logger.error(f"오류 세부 정보: {str(e)}")
            return None

def main():
    """메인 함수"""
    # 토크나이저 초기화
    tokenizer = MedicalTokenizer()
    
    # 질병 데이터 처리
    logger.info(f"질병 데이터 처리 시작: {DISEASE_CSV_PATH}")
    df_processed = tokenizer.process_disease_csv()
    
    if df_processed is not None:
        logger.info(f"처리 완료: {len(df_processed)}개 레코드")
        
        # 토큰 통계 분석 (추가)
        unique_tokens = set()
        token_freq = {}
        
        for _, row in df_processed.iterrows():
            if 'tokens' in row and isinstance(row['tokens'], list):
                for token in row['tokens']:
                    unique_tokens.add(token)
                    if token not in token_freq:
                        token_freq[token] = 0
                    token_freq[token] += 1
        
        logger.info(f"총 고유 토큰 수: {len(unique_tokens)}")
        
        # 가장 빈번한 토큰 상위 10개 출력
        top_tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        for token, freq in top_tokens:
            logger.info(f"빈번한 토큰: '{token}' ({freq}회)")
    else:
        logger.error("처리 실패")

if __name__ == "__main__":
    main()