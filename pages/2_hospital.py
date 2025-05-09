import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
import requests
from math import radians, sin, cos, sqrt, atan2
from dotenv import load_dotenv
import os

# ─────────────────────────────
# 환경 설정 및 데이터 불러오기
# ─────────────────────────────
load_dotenv()
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
df = pd.read_csv("pages/hospital_with_latlon.csv")

# ─────────────────────────────
# 유틸 함수
# ─────────────────────────────
def get_lat_lon(address):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    params = {"query": address}
    res = requests.get(url, headers=headers, params=params)
    res_json = res.json()
    if res_json["documents"]:
        x = float(res_json["documents"][0]["x"])
        y = float(res_json["documents"][0]["y"])
        return y, x
    return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

def match_exact_departments(treatment, selected_depts):
    if pd.isna(treatment):
        return False
    dept_list = [s.strip() for s in treatment.split(",")]
    return any(dept in dept_list for dept in selected_depts)

# ─────────────────────────────
# 지도 및 리스트 출력 함수
# ─────────────────────────────
def show_map_and_list(center, radius, df_filtered):
    center_lat, center_lon = center

    df_filtered["distance"] = df_filtered.apply(
        lambda row: haversine(center_lat, center_lon, row["lat"], row["lon"]), axis=1
    )
    df_nearby = df_filtered[df_filtered["distance"] <= radius].sort_values("distance")

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
    folium.Marker(
        [center_lat, center_lon],
        icon=folium.Icon(color="red", icon="info-sign"),
        tooltip="선택 위치",
        popup=folium.Popup("중심 위치", max_width=300)
    ).add_to(m)

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
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color="blue", icon="plus-sign")
        ).add_to(marker_cluster)

    col_map, col_list = st.columns([2, 1])
    with col_map:
        st_folium(m, width=700, height=600)
    with col_list:
        st.markdown("### 📋 병원 목록")
        if df_nearby.empty:
            st.info("🔎 조건에 맞는 병원이 없습니다.")
        for _, row in df_nearby.iterrows():
            st.markdown(f"**🏥 {row['hospital_name']}**")
            st.markdown(f"- 진료과: {row['treatment']}")
            st.markdown(f"- 주소: {row['address']}")
            st.markdown(f"- 거리: {row['distance']:.2f} km")
            st.markdown("---")

# ─────────────────────────────
# 주소 입력 처리
# ─────────────────────────────
def render_address_input(df_filtered, radius):
    address = st.text_input("도로명 주소 입력", "서울특별시 광진구 능동로 120")
    if address:
        center = get_lat_lon(address)
        if center:
            show_map_and_list(center, radius, df_filtered)
        else:
            st.warning("❌ 주소를 찾을 수 없습니다.")

# ─────────────────────────────
# GPS 위치 처리 + 재요청
# ─────────────────────────────
def render_gps_location(df_filtered, radius):
    if "gps_location" not in st.session_state:
        with st.spinner("📡 위치 정보를 가져오는 중입니다..."):
            st.session_state["gps_location"] = get_geolocation()

    location = st.session_state.get("gps_location")
    coords = location.get("coords") if location else None

    st.write("📌 location 반환값:", location)  # 디버깅용

    if coords and coords.get("latitude") and coords.get("longitude"):
        lat = coords["latitude"]
        lon = coords["longitude"]
        acc = coords.get("accuracy", 9999)

        if acc > 1000:
            st.warning(f"⚠️ 현재 위치 정확도가 낮습니다. (±{int(acc)}m)")

        show_map_and_list((lat, lon), radius, df_filtered)
    else:
        st.warning("⚠️ 위치 권한을 허용해주세요.")
        if st.button("🔄 위치 다시 요청"):
            with st.spinner("📡 위치 재요청 중..."):
                st.session_state["gps_location"] = get_geolocation()

# ─────────────────────────────
# Streamlit UI
# ─────────────────────────────
st.set_page_config(page_title="병원 지도 서비스", layout="wide")
st.title("🏥 병원 위치 시각화 서비스")

radius = st.slider("📏 반경 (km)", 0.1, 5.0, 1.0, 0.1)

# 진료과 필터
all_departments = set()
df["treatment"].dropna().apply(lambda t: all_departments.update([s.strip() for s in t.split(",")]))
departments = sorted(list(all_departments))
selected_depts = st.multiselect("진료과 필터", departments)

df_filtered = df.copy()
if selected_depts:
    df_filtered = df_filtered[df_filtered["treatment"].apply(lambda t: match_exact_departments(t, selected_depts))]

# 위치 입력 방식
method = st.radio("📍 위치 입력 방식", ["주소 입력", "현재 위치(GPS)"], horizontal=True)

if method == "주소 입력":
    render_address_input(df_filtered, radius)
elif method == "현재 위치(GPS)":
    render_gps_location(df_filtered, radius)
