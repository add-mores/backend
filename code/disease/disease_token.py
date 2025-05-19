import os
import re
import pandas as pd
import json
import numpy as np
from konlpy.tag import Okt, Mecab
import logging
from datetime import datetime
import argparse

# 파일 경로 설정 (파일 위치만 조절하면 됨)
DISEASE_CSV_PATH = "/Users/jacob/Desktop/token/disease_data.csv"  # 질병 데이터 CSV 파일 경로
MED_TERMS_CSV_PATH = "/Users/jacob/Desktop/token/medical_terms_cleaned.csv"  # 의학 용어 CSV 파일 경로
DISEASE_TERMS_CSV_PATH = "/Users/jacob/Desktop/token/medical_terms.csv"  # 질병 용어 CSV 파일 경로 (새로 추가)
OUTPUT_CSV_PATH = "/Users/jacob/Desktop/token/processed_disease_data.csv"  # 출력 CSV 파일 경로
STOPWORDS_LOG_PATH = "/Users/jacob/Desktop/token/remaining_stopwords.txt"  # 남아있는 불용어 로그 파일 경로

# 수동 정의한 복합 질병명 목록
MANUAL_COMPOUND_DISEASE_NAMES = [
    "급성 기관지염", "만성 기관지염", "급성 폐렴", "만성 폐렴", "급성 위염", "만성 위염",
    "급성 충수염", "만성 충수염", "담낭염", "담낭 결석", "신장 결석", "요로 결석",
    "알레르기성 비염", "알레르기 비염", "알레르기 결막염", "알레르기 피부염",
    "당뇨병", "제1형 당뇨병", "제2형 당뇨병", "인슐린 의존성 당뇨병", "인슐린 비의존성 당뇨병",
    "고혈압", "원발성 고혈압", "이차성 고혈압", "폐 고혈압", "문맥 고혈압",
    "관상동맥 질환", "관상동맥 경화증", "심근 경색", "심근 허혈", "심장 판막 질환",
    "뇌졸중", "뇌출혈", "뇌혈관 질환", "일과성 허혈 발작", "지주막하 출혈",
    "위식도 역류 질환", "위식도 역류", "위궤양", "십이지장 궤양", "소화성 궤양",
    "크론병", "궤양성 대장염", "과민성 대장 증후군", "대장 용종", "대장 폴립",
    "갑상선 기능 항진증", "갑상선 기능 저하증", "갑상선염", "갑상선 결절", "갑상선 암",
    "류마티스 관절염", "골관절염", "강직성 척추염", "통풍", "섬유근통",
    "편두통", "긴장성 두통", "군발성 두통", "만성 두통", "삼차신경통",
    "천식", "만성 폐쇄성 폐질환", "간질성 폐질환", "폐색전증", "폐결핵",
    "간염", "간경화", "지방간", "알코올성 간질환", "비알코올성 지방간염",
    "우울증", "불안 장애", "공황 장애", "강박 장애", "외상후 스트레스 장애",
    "알츠하이머병", "파킨슨병", "다발성 경화증", "근위축성 측삭 경화증", "헌팅턴병"
]

# 로깅 설정
log_filename = f"tokenizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler(log_filename, encoding='utf-8'),
                       logging.StreamHandler()
                   ])
logger = logging.getLogger(__name__)

# medical_terms.csv에서 복합 질병명 로드
def load_disease_names_from_csv():
    """medical_terms.csv에서 질병명을 로드하여 복합 질병명 목록에 추가"""
    compound_disease_names = set(MANUAL_COMPOUND_DISEASE_NAMES)  # 중복 제거를 위해 set 사용
    
    try:
        if os.path.exists(DISEASE_TERMS_CSV_PATH):
            # CSV 파일 로드
            df = pd.read_csv(DISEASE_TERMS_CSV_PATH, encoding='utf-8')
            
            if '질환명' in df.columns:
                # 질환명 컬럼에서 질병명 추출
                for _, row in df.iterrows():
                    disease_name = str(row['질환명']).strip()
                    
                    # 복합 질병명 (공백 포함)인 경우만 추가
                    if disease_name and ' ' in disease_name:
                        compound_disease_names.add(disease_name)
                    
                    # '-' 포함 질병명도 추가 (예: "크론-병")
                    elif disease_name and '-' in disease_name:
                        # '-'를 공백으로 바꾸어 추가
                        compound_disease_names.add(disease_name.replace('-', ' '))
                
                logger.info(f"medical_terms.csv에서 {len(compound_disease_names) - len(MANUAL_COMPOUND_DISEASE_NAMES)}개의 추가 복합 질병명 로드")
            else:
                logger.warning("medical_terms.csv에 '질환명' 컬럼이 없습니다.")
        else:
            logger.warning(f"질병 용어 파일이 없습니다: {DISEASE_TERMS_CSV_PATH}")
        
    except Exception as e:
        logger.error(f"질병명 로드 오류: {e}")
    
    return list(compound_disease_names)

# medical_terms.csv에서 복합 질병명 로드
COMPOUND_DISEASE_NAMES = load_disease_names_from_csv()

class MedicalTokenizer:
    """
    의학 텍스트 토큰화 모듈 (CSV 기반 버전)
    - 의학 용어 사전 활용
    - 불용어 처리 개선
    - 화이트리스트 확장
    - 복합 질병명 보존
    - 디버깅 기능 추가
    """
    
    def __init__(self, 
                med_terms_file=MED_TERMS_CSV_PATH,
                use_mecab=False,
                debug_mode=False):
        """
        토크나이저 초기화
        Args:
            med_terms_file: 의학 용어 CSV 파일 경로
            use_mecab: Mecab 형태소 분석기 사용 여부 (False면 Okt 사용)
            debug_mode: 디버깅 모드 활성화 여부
        """
        self.debug_mode = debug_mode
        
        # 형태소 분석기 초기화
        self.tokenizer = Mecab() if use_mecab else Okt()
        logger.info(f"형태소 분석기 초기화: {'Mecab' if use_mecab else 'Okt'}")
        
        # 화이트리스트: 의학 용어, 특수 단어 등 보존할 단어 목록
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
        
        # 불용어: 제거할 단어 목록
        self.stopwords = set([
            # 일반 한국어 불용어
            "있다", "없다", "되다", "하다", "이다", "같다", "때문", "따라서",
            "그러나", "하지만", "또한", "등", "및", "에서", "으로", "이나",
            "그리고", "또는", "경우", "통해", "위해", "대해", "이런", "그런",
        ])
        
        # PDF 문서에서 추출한 불용어 추가
        self.stopwords.update(self.load_pdf_stopwords())
        
        # 복합 명사: 함께 처리할 단어 목록 (복합 질병명 추가)
        self.compound_nouns = set([
            "급성 폐렴", "만성 두통", "관절염", "심근경색", "뇌출혈", 
            "위장 장애", "고혈압", "저혈압", "당뇨병", "갑상선 기능 저하증"
        ])
        
        # 복합 질병명 추가
        for disease_name in COMPOUND_DISEASE_NAMES:
            if disease_name and isinstance(disease_name, str):
                self.compound_nouns.add(disease_name)
                self.whitelist.add(disease_name)
        
        # CSV 파일에서 의학 용어 로드
        self.load_medical_terms_from_csv(med_terms_file)
        
        # 화이트리스트와 불용어 목록 충돌 확인 (디버깅)
        conflicts = self.whitelist.intersection(self.stopwords)
        if conflicts:
            logger.warning(f"화이트리스트와 불용어 목록 사이에 {len(conflicts)}개의 충돌이 있습니다.")
            if self.debug_mode:
                logger.debug(f"충돌 목록 (최대 20개): {list(conflicts)[:20]}")
            
            # 충돌 해결: 화이트리스트 우선
            for term in conflicts:
                self.stopwords.remove(term)
                if self.debug_mode:
                    logger.debug(f"불용어 목록에서 제거됨: '{term}'")
        
        logger.info(f"화이트리스트 크기: {len(self.whitelist)}")
        logger.info(f"불용어 크기: {len(self.stopwords)}")
        logger.info(f"복합 명사 크기: {len(self.compound_nouns)}")
        
        # 디버깅용 변수
        self.remaining_stopwords = {}  # 각 불용어별 출현 횟수 추적
    
    def load_medical_terms_from_csv(self, med_terms_file):
        """CSV에서 의학 용어 로드"""
        try:
            if not os.path.exists(med_terms_file):
                logger.warning(f"의학 용어 파일이 없습니다: {med_terms_file}")
                return
                
            # 의학 용어 CSV 로드
            med_terms_df = pd.read_csv(med_terms_file, encoding='utf-8')
            logger.info(f"의학 용어 CSV 로드 완료: {len(med_terms_df)}개 용어")
            
            # 필요한 컬럼 확인
            required_columns = ['medterm', 'medterm_eng', 'syn']
            missing_columns = [col for col in required_columns if col not in med_terms_df.columns]
            if missing_columns:
                logger.warning(f"의학 용어 CSV에 필요한 컬럼이 없습니다: {missing_columns}")
            
            # 의학 용어를 화이트리스트에 추가
            for _, row in med_terms_df.iterrows():
                # 한글 의학 용어 추가
                if 'medterm' in row and pd.notna(row['medterm']):
                    term = str(row['medterm']).strip()
                    if term and len(term) > 1:
                        self.whitelist.add(term)
                
                # 영문 의학 용어 추가
                if 'medterm_eng' in row and pd.notna(row['medterm_eng']):
                    eng_term = str(row['medterm_eng']).strip()
                    if eng_term and len(eng_term) > 1:
                        self.whitelist.add(eng_term)
                
                # 동의어 추가
                if 'syn' in row and pd.notna(row['syn']):
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
            
            logger.info(f"의학 용어 CSV에서 {len(self.whitelist)}개 용어를 화이트리스트에 추가")
            
        except Exception as e:
            logger.error(f"의학 용어 로드 오류: {e}")
    
    def load_pdf_stopwords(self):
        """PDF 문서에서 추출한 불용어 목록"""
        # PDF 문서에서 추출한 추가 불용어 목록
        pdf_stopwords = [
            "발생", "시행", "수술", "사용", "통증", "가장", "동반", "부위", "때문", "약물",
            "대부분", "기능", "통해", "대한", "다른", "위해", "감소", "주로", "효과", "가지",
            "제거", "조직", "거나", "또한", "이용", "정상", "여러", "진행", "의미",
            "지속", "장애", "수도", "요법", "투여", "증가",
            "유발", "매우", "따라서", "변화", "위치", "도움", "절제", "관찰", "고려",
            "대개", "이후", "부분", "조절", "일부", "유지", "반응",
            "전신", "시작", "호전", "특징", "복용", "항생제", "신경",
            "초기", "세포", "형태", "포함", "문제", "합병증", "사람", "곤란", "개월", "최근",
            "신체", "급성", "장기", "발견", "예방", "사이", "치료법",
            "동안", "완화", "보통", "재발", "기간", "스테로이드", "반복", "대해", "관련",
            "경과", "보고", "주위", "모든", "결과", "교정", "모양", "만성",
            "전체", "하나", "다음", "주변", "가능성", "조기", "발병", "성인", "복부",
            "회복", "면역", "크게", "결정", "여성", "다시", "형성", "위험", "물질",
            "세균", "보존", "호소", "주사", "빈도", "시도", "현재", "비교", "해당",
            "처음", "약제", "존재", "양상", "이내", "필요", "비정상", "선택", "과정",
            "침범", "보이", "의심", "거의", "방사선", "생기", "자연", "시술", "시기",
            "활동", "목적", "공급", "소실", "영향", "부작용", "분류", "자체",
            "호르몬", "악화", "자주", "이전", "사망", "분비", "삽입", "제한", "이식", "정신",
            "안정", "현상", "바이러스", "점막", "직접", "정맥", "여부", "이외", "작용", "기침",
            "국소", "행동", "우선", "연령", "신장", "의사", "노출", "마비", "절개"
        ]
        return pdf_stopwords
    
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
        특수 패턴 추출 (혈액형, 의학 용어 등)
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
    
    def is_valid_token(self, token):
        """
        유효한 토큰인지 확인
        Args:
            token: 확인할 토큰
        Returns:
            유효 여부
        """
        # 디버깅: 불용어 목록에 있는데 화이트리스트에도 있는 단어 확인
        if self.debug_mode and token in self.stopwords and token in self.whitelist:
            logger.debug(f"토큰 '{token}'은 불용어이지만 화이트리스트에도 있어 보존됩니다.")
        
        # 화이트리스트에 있으면 유효
        if token in self.whitelist:
            return True
            
        # 복합 명사에 있으면 유효
        if token in self.compound_nouns:
            return True
            
        # 불용어에 있으면 유효하지 않음
        if token in self.stopwords:
            # 디버깅: 불용어 등장 추적
            if self.debug_mode:
                if token not in self.remaining_stopwords:
                    self.remaining_stopwords[token] = 0
                self.remaining_stopwords[token] += 1
            return False
            
        # 길이가 1이면 유효하지 않음 (의미없는 토큰)
        if len(token) < 2:
            return False
            
        # 한글, 영문, 숫자만 포함된 토큰만 유효
        return bool(re.fullmatch(r"[\w가-힣ㄱ-ㅎㅏ-ㅣ]+", token))
    
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
    
    def tokenize(self, text):
        """
        텍스트 토큰화 메인 함수
        Args:
            text: 입력 텍스트
        Returns:
            토큰화된 단어 목록
        """
        # 빈 텍스트 처리
        if not text or not isinstance(text, str):
            return []
            
        # 텍스트 전처리
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return []
        
        # 화이트리스트 토큰 추출
        whitelist_tokens = []
        for token in self.whitelist:
            if token in cleaned_text and token not in whitelist_tokens:
                whitelist_tokens.append(token)
        
        # 특수 패턴 토큰 추출
        special_tokens = self.extract_special_patterns(cleaned_text)
        
        # 의학 용어 토큰 추출
        medical_tokens = self.extract_medical_terms(cleaned_text)
        
        # 추출된 토큰들을 원본 텍스트에서 제거 (중복 추출 방지)
        temp = cleaned_text
        for w in whitelist_tokens + special_tokens + medical_tokens:
            # 정규식 패턴으로 단어 경계 지정하여 제거
            temp = re.sub(rf"\b{re.escape(w)}\b", " ", temp)
            temp = re.sub(r"\s+", " ", temp).strip()
        
        # 남은 텍스트에서 명사 추출
        nouns = []
        if self.tokenizer:
            nouns = self.tokenizer.nouns(temp)
        
        # 모든 토큰 합치고 중복 제거 및 불용어 제거
        all_tokens = whitelist_tokens + special_tokens + medical_tokens + nouns
        final_tokens = []
        
        for token in all_tokens:
            if token and token not in final_tokens and self.is_valid_token(token):
                final_tokens.append(token)
                
        return final_tokens
    
    def process_disease_csv(self, disease_csv_path, output_csv_path=None):
        """
        질병 데이터 CSV 토큰화 처리
        Args:
            disease_csv_path: 질병 데이터 CSV 파일 경로
            output_csv_path: 출력 CSV 파일 경로 (None이면 'processed_disease_data.csv' 사용)
        Returns:
            토큰화된 결과가 저장된 DataFrame
        """
        try:
            # CSV 파일 존재 확인
            if not os.path.exists(disease_csv_path):
                logger.error(f"질병 데이터 CSV 파일이 없습니다: {disease_csv_path}")
                return None
                
            # CSV에서 질병 데이터 로드
            disease_df = pd.read_csv(disease_csv_path, encoding='utf-8')
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
            
            # 디버깅용: 남아있는 불용어 추적
            all_remaining_stopwords = {}
            
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
                    
                    # 각 필드 토큰화
                    def_tokens = self.tokenize(definition)
                    symp_tokens = self.tokenize(symptoms)
                    #therapy_tokens = self.tokenize(therapy)
                    
                    # 모든 토큰 결합 및 중복 제거
                    all_tokens = list(set(def_tokens + symp_tokens))
                    
                    # 정의 토큰 문자열 생성 (def_k)
                    def_k = " ".join(def_tokens)
                    
                    # 디버깅: 남아있는 불용어 확인
                    if self.debug_mode:
                        for token in all_tokens:
                            if token in self.stopwords:
                                if token not in all_remaining_stopwords:
                                    all_remaining_stopwords[token] = 0
                                all_remaining_stopwords[token] += 1
                                logger.debug(f"불용어 '{token}'이 질병 '{disnm_ko}'의 토큰 목록에 남아 있습니다.")
                    
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
            
            # DataFrame 생성
            df_processed = pd.DataFrame(processed_records)
            
            # 토큰에 남아 있는 불용어 로깅
            if self.debug_mode and all_remaining_stopwords:
                # 빈도 기준 내림차순 정렬
                sorted_stopwords = sorted(all_remaining_stopwords.items(), key=lambda x: x[1], reverse=True)
                logger.warning(f"토큰에 남아 있는 불용어 수: {len(all_remaining_stopwords)}")
                
                # 상위 20개 불용어 로깅
                top_stopwords = sorted_stopwords[:20]
                for word, count in top_stopwords:
                    logger.warning(f"불용어 '{word}': {count}회 출현")
                
                # 남아있는 불용어 목록 파일 저장
                with open(STOPWORDS_LOG_PATH, 'w', encoding='utf-8') as f:
                    f.write(f"남아있는 불용어 목록 (총 {len(sorted_stopwords)}개)\n")
                    f.write(f"형식: 불용어 [출현 횟수]\n\n")
                    for word, count in sorted_stopwords:
                        f.write(f"{word} [{count}]\n")
                logger.info(f"남아있는 불용어 목록을 {STOPWORDS_LOG_PATH}에 저장했습니다.")
                
                # 화이트리스트와 충돌하는 불용어 확인
                conflicts = [word for word in all_remaining_stopwords if word in self.whitelist]
                if conflicts:
                    logger.warning(f"화이트리스트와 충돌하는 불용어: {len(conflicts)}개")
                    logger.warning(f"예시: {conflicts[:10]}")
                
                # 불용어 목록 초기화 (메모리 해제)
                self.remaining_stopwords = {}
            
            # tokens 열을 JSON 문자열로 변환 (CSV 저장 시 필요)
            df_processed['tokens_json'] = df_processed['tokens'].apply(lambda x: json.dumps(x, ensure_ascii=False))
            
            # 결과 CSV 저장
            if output_csv_path is None:
                output_csv_path = OUTPUT_CSV_PATH
                
            # CSV 파일로 저장
            df_processed.to_csv(output_csv_path, index=False, encoding='utf-8')
            logger.info(f"✅ {output_csv_path}에 {len(df_processed)}개 레코드 저장 완료")
            
            # tokens_json 열 제거하고 원래 tokens 열만 반환
            df_processed = df_processed.drop(columns=['tokens_json'])
            
            return df_processed
            
        except Exception as e:
            logger.error(f"질병 데이터 처리 오류: {e}")
            return None

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='질병 데이터 토큰화 처리')
    parser.add_argument('--disease_csv', type=str, default=DISEASE_CSV_PATH, 
                        help='질병 데이터 CSV 파일 경로')
    parser.add_argument('--med_terms_csv', type=str, default=MED_TERMS_CSV_PATH, 
                        help='의학 용어 CSV 파일 경로')
    parser.add_argument('--output_csv', type=str, default=OUTPUT_CSV_PATH, 
                        help='출력 CSV 파일 경로')
    parser.add_argument('--use_mecab', action='store_true', 
                        help='Mecab 형태소 분석기 사용 (기본: Okt)')
    parser.add_argument('--debug', action='store_true', 
                        help='디버깅 모드 활성화')
    args = parser.parse_args()
    
    # 디버깅 모드 활성화 시 로그 레벨 변경
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("디버깅 모드가 활성화되었습니다.")
    
    # 토크나이저 초기화
    tokenizer = MedicalTokenizer(
        med_terms_file=args.med_terms_csv,
        use_mecab=args.use_mecab,
        debug_mode=args.debug
    )
    
    # 질병 데이터 처리
    logger.info(f"질병 데이터 처리 시작: {args.disease_csv}")
    df_processed = tokenizer.process_disease_csv(args.disease_csv, args.output_csv)
    
    if df_processed is not None:
        logger.info(f"처리 완료: {len(df_processed)}개 레코드")
        
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
        else:
            logger.warning("토큰 통계를 계산할 수 없습니다.")
    else:
        logger.error("처리 실패")

if __name__ == "__main__":
    main()