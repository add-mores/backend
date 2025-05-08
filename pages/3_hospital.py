import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import requests
from math import radians, sin, cos, sqrt, atan2
from dotenv import load_dotenv
import os

# 📌 환경 변수 불러오기
load_dotenv()
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}

# 📌 주소 → 위경도 변환 함수
def get_lat_lon(address):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    params = {"query": address}
    res = requests.get(url, headers=headers, params=params)
    res_json = res.json()
    if res_json["documents"]:
        x = float(res_json["documents"][0]["x"])
        y = float(res_json["documents"][0]["y"])
        return y, x
    return None, None

# 📌 거리 계산 함수 (Haversine)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# 📌 병원 데이터 불러오기
df = pd.read_csv("pages/hospital_with_latlon.csv")

# 📌 Streamlit 설정 및 iframe 높이 고정
st.set_page_config(page_title="병원 지도 서비스", layout="wide")
st.markdown("""
<style>
iframe {
    min-height: 600px !important;
    max-height: 600px !important;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# 📌 상태 변수 초기화
if "map_shown" not in st.session_state:
    st.session_state["map_shown"] = False
if "last_address" not in st.session_state:
    st.session_state["last_address"] = ""

st.title("🏥 병원 위치 시각화 서비스")

# 📌 진료과 목록 추출 및 체크박스 필터링
all_departments = set()
df["treatment"].dropna().apply(lambda t: all_departments.update([s.strip() for s in t.split(",")]))
departments = sorted(list(all_departments))
selected_depts = st.multiselect("필터링할 진료과를 선택하세요", departments)

df_filtered = df.copy()
if selected_depts:
    df_filtered = df[df["treatment"].apply(
        lambda t: any(dept in t for dept in selected_depts) if pd.notna(t) else False
    )]

# 📌 주소 입력
address = st.text_input("도로명 주소를 입력하세요", value="서울특별시 광진구 능동로 120")

# 📌 주소가 바뀌면 지도 리셋
if address != st.session_state["last_address"]:
    st.session_state["map_shown"] = False

# 📌 지도 보기 / 숨기기 버튼
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("지도 보기"):
        st.session_state["map_shown"] = True
        st.session_state["last_address"] = address
with col2:
    if st.button("지도 숨기기"):
        st.session_state["map_shown"] = False

# 📌 지도 출력 조건
if st.session_state["map_shown"]:
    center_lat, center_lon = get_lat_lon(address)

    if center_lat is None:
        st.error("❌ 주소를 찾을 수 없습니다.")
    else:
        # 📌 거리 필터링
        df_filtered["distance"] = df_filtered.apply(
            lambda row: haversine(center_lat, center_lon, row["lat"], row["lon"]),
            axis=1
        )
        df_nearby = df_filtered[df_filtered["distance"] <= 1.0]

        # 📌 지도 생성
        m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

        # 중심 마커
        folium.Marker(
            [center_lat, center_lon],
            icon=folium.Icon(color="red", icon="info-sign"),
            tooltip="입력 위치",
            popup=folium.Popup(f"<div style='white-space: nowrap; font-size: 14px;'>{address}</div>")
        ).add_to(m)

        # 마커 클러스터
        marker_cluster = MarkerCluster().add_to(m)

        for _, row in df_nearby.iterrows():
            popup_html = f"""
            <div style="width: 220px;">
                <strong>{row['hospital_name']}</strong><br>
                <ul style="padding-left: 18px; margin: 6px 0;">
                    {''.join(f"<li>{s.strip()}</li>" for s in str(row['treatment']).split(','))}
                </ul>
                <p>{row['address']}</p>
            </div>
            """
            folium.Marker(
                [row["lat"], row["lon"]],
                tooltip=row["hospital_name"],
                popup=folium.Popup(popup_html, max_width=300, min_width=150),
                icon=folium.Icon(color="blue", icon="plus-sign")
            ).add_to(marker_cluster)

        # 📌 지도 출력
        st_data = st_folium(m, width=700, height=600)

        # 📌 병원 리스트 출력
        st.markdown("### 📋 반경 1km 병원 목록")
        for _, row in df_nearby.iterrows():
            st.markdown(f"**🏥 {row['hospital_name']}**")
            st.markdown(f"- 진료과목: {row['treatment']}")
            st.markdown(f"- 주소: {row['address']}")
