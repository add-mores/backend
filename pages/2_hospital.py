import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
from math import radians, sin, cos, sqrt, atan2
import requests
from dotenv import load_dotenv
import os

# ─────────────────────── 설정 및 데이터 로드 ───────────────────────
load_dotenv()
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
df = pd.read_csv("pages/hospital_with_latlon.csv")

# ─────────────────────── 유틸 함수 ───────────────────────
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

# ─────────────────────── 지도 + 리스트 출력 ───────────────────────
def show_map_and_list(center, radius, df_filtered):
    center_lat, center_lon = center
    df_filtered["distance"] = df_filtered.apply(
        lambda row: haversine(center_lat, center_lon, row["lat"], row["lon"]), axis=1
    )
    df_nearby = df_filtered[df_filtered["distance"] <= radius].sort_values("distance").reset_index(drop=True)

    # 지도 중심 설정 (선택한 병원이 있으면 해당 위치로)
    focused = st.session_state.get("focused_location", center)

    m = folium.Map(location=focused, zoom_start=16)
    folium.Marker(center, icon=folium.Icon(color="red", icon="info-sign"), tooltip="선택 위치").add_to(m)
    cluster = MarkerCluster().add_to(m)

    for _, row in df_nearby.iterrows():
        popup_html = f"""
        <strong style='color:black'>{row['hospital_name']}</strong><br>
        <span style='color:black'>{row['address']}<br>{row['treatment']}</span>
        """
        marker = folium.Marker(
            [row["lat"], row["lon"]],
            tooltip=row["hospital_name"],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color="blue", icon="plus-sign")
        )
        marker.add_to(cluster)

    # 지도 + 리스트 레이아웃
    map_col, list_col = st.columns([3, 2])
    with map_col:
        st_folium(m, width=700, height=600)

    with list_col:
        st.markdown("### 📋 병원 목록")

        if df_nearby.empty:
            st.info("❌ 조건에 맞는 병원이 없습니다.")
            return

        # 보여줄 병원 개수 제한
        if "visible_count" not in st.session_state:
            st.session_state.visible_count = 3

        visible = st.session_state.visible_count
        total = len(df_nearby)
        hospitals_to_show = df_nearby.iloc[:visible]

        for idx, row in hospitals_to_show.iterrows():
            hosp_id = f"hospital_{idx}"
            is_expanded = st.session_state.get(hosp_id, False)

            # 병원 이름 버튼 (클릭 시 지도 이동)
            if st.button(f"📍 {row['hospital_name']} ({row['distance']:.2f} km)", key=hosp_id + "_click"):
                st.session_state["focused_location"] = (row["lat"], row["lon"])
                st.rerun()

            # 상세 정보 (더보기 클릭 여부)
            if is_expanded:
                st.markdown(f"""
                <div style="background-color:white;padding:10px;border-radius:10px;margin-bottom:8px;">
                    <strong style="color:black">{row['hospital_name']}</strong><br>
                    <span style="font-size: 13px; color: black;">
                    주소: {row['address']}<br>
                    진료과: {row['treatment']}<br>
                    거리: {row['distance']:.2f} km
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color:white;padding:8px;border-radius:10px;margin-bottom:8px;">
                    <strong style="color:black">{row['hospital_name']}</strong><br>
                    <span style="font-size: 13px; color: black;">
                    거리: {row['distance']:.2f} km
                    </span>
                </div>
                """, unsafe_allow_html=True)

            if st.button("➕ 더보기" if not is_expanded else "➖ 닫기", key=hosp_id + "_toggle"):
                st.session_state[hosp_id] = not is_expanded
                st.rerun()

        # 더보기 버튼 (카드 3개씩 추가 표시)
        if visible < total:
            if st.button("📄 병원 더보기"):
                st.session_state.visible_count += 3
                st.rerun()

# ─────────────────────── 주소 입력 처리 ───────────────────────
def render_address_input(df_filtered, radius):
    address = st.text_input("도로명 주소 입력", "서울특별시 광진구 능동로 120")
    if address:
        center = get_lat_lon(address)
        if center:
            st.session_state["focused_location"] = center
            show_map_and_list(center, radius, df_filtered)
        else:
            st.warning("❌ 주소를 찾을 수 없습니다.")

# ─────────────────────── GPS 처리 ───────────────────────
def render_gps_location(df_filtered, radius):
    if "gps_location" not in st.session_state:
        with st.spinner("📡 위치 정보를 가져오는 중입니다..."):
            st.session_state["gps_location"] = get_geolocation()

    location = st.session_state.get("gps_location")
    coords = location.get("coords") if location else None

    if coords and coords.get("latitude") and coords.get("longitude"):
        lat = coords["latitude"]
        lon = coords["longitude"]
        acc = coords.get("accuracy", 9999)

        st.info(f"📍 현재 위치 정확도: ±{int(acc)}m")

        if acc <= 100:
            st.session_state["focused_location"] = (lat, lon)
            show_map_and_list((lat, lon), radius, df_filtered)
        else:
            st.warning("⚠️ 현재 위치 정확도가 낮습니다. 주소 입력을 권장합니다.")
            if st.button("📍 주소 입력으로 전환"):
                st.session_state["location_method"] = "주소 입력"
                st.rerun()
    else:
        st.warning("⚠️ 위치 정보를 가져올 수 없습니다.")

    if st.button("🔄 위치 다시 요청"):
        st.session_state["gps_location"] = get_geolocation()
        st.rerun()

# ─────────────────────── Streamlit UI ───────────────────────
st.set_page_config(page_title="병원 위치 시각화", layout="wide")
st.markdown("<h1 style='text-align: center;'>🏥 병원 위치 시각화 서비스</h1>", unsafe_allow_html=True)

# 상태 초기화
if "location_method" not in st.session_state:
    st.session_state["location_method"] = "현재 위치(GPS)"
if "focused_location" not in st.session_state:
    st.session_state["focused_location"] = (37.5665, 126.9780)  # 기본: 서울 시청

# 위치 입력 방식 선택
ui_method = st.radio("위치 입력 방식", ["현재 위치(GPS)", "주소 입력"],
                     index=0 if st.session_state["location_method"] == "현재 위치(GPS)" else 1,
                     horizontal=True)

if ui_method != st.session_state["location_method"]:
    st.session_state["location_method"] = ui_method
    st.rerun()

# 반경 및 진료과 필터
col1, col2 = st.columns(2)
with col1:
    radius = st.slider("📏 반경 (km)", 0.1, 5.0, 1.0, 0.1)
with col2:
    all_departments = set()
    df["treatment"].dropna().apply(lambda t: all_departments.update([s.strip() for s in t.split(",")]))
    selected_depts = st.multiselect("진료과 필터", sorted(all_departments), placeholder="진료과 선택")

df_filtered = df.copy()
if selected_depts:
    df_filtered = df_filtered[df_filtered["treatment"].apply(lambda t: match_exact_departments(t, selected_depts))]

# 위치 기반 병원 표시
if st.session_state["location_method"] == "현재 위치(GPS)":
    render_gps_location(df_filtered, radius)
else:
    render_address_input(df_filtered, radius)

