import requests
from bs4 import BeautifulSoup
import csv
import time
import random

# CSV 파일 생성
csv_filename = 'medical_terms.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['meditem', 'syn', 'des']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    # dictId 1부터 5016까지 순회
    for dict_id in range(1, 5017):
        # URL 생성
        url = f"https://www.amc.seoul.kr/asan/healthinfo/easymediterm/easyMediTermDetail.do?dictId={dict_id}"
        
        try:
            # 서버에 과부하를 주지 않기 위해 요청 간격 조절
            time.sleep(random.uniform(0.5, 1.5))
            
            # 요청 보내기
            response = requests.get(url)
            
            # 응답이 성공적인지 확인
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 의학용어 추출 (h1 태그)
                meditem_elem = soup.find('h1')
                meditem = meditem_elem.text.strip() if meditem_elem else ""
                
                # 동의어 추출 (contBox 클래스 내에서 dt가 '동의어'인 dd 태그)
                syn = ""
                contbox = soup.find('div', class_='contBox')
                if contbox:
                    dl_elem = contbox.find('dl')
                    if dl_elem:
                        dt_tags = dl_elem.find_all('dt')
                        dd_tags = dl_elem.find_all('dd')
                        for i, dt in enumerate(dt_tags):
                            if '동의어' in dt.text and i < len(dd_tags):
                                syn = dd_tags[i].text.strip()
                
                # 정의 추출 (description 클래스 내의 dd 태그)
                des = ""
                description = soup.find('div', class_='description')
                if description:
                    dd_elem = description.find('dd')
                    if dd_elem:
                        des = dd_elem.text.strip()
                
                # CSV에 데이터 쓰기
                writer.writerow({
                    'meditem': meditem,
                    'syn': syn,
                    'des': des
                })
                
                # 진행상황 출력
                if dict_id % 100 == 0:
                    print(f"Processed {dict_id}/5016 pages")
                    
            else:
                print(f"Failed to retrieve page for dictId={dict_id}, status code: {response.status_code}")
                
        except Exception as e:
            print(f"Error processing dictId={dict_id}: {str(e)}")
            # 오류가 발생해도 계속 진행
            continue

print(f"Data collection completed. Results saved to {csv_filename}")
