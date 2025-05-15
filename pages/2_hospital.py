import streamlit as st
import pandas as pd
import numpy as np
import folium
from sqlalchemy import create_engine
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from streamlit_js_eval import get_geolocation
import requests
import psycopg2
from math import radians, sin, cos, sqrt, atan2
from dotenv import load_dotenv
import os

# ─────────────── 설정 ───────────────
load_dotenv()
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}

# ─────────────── DB 연결 및 병원 데이터 로딩 ───────────────

def load_hospital_data():
    engine = create_engine(DATABASE_URL)
    query = """
        SELECT hos_nm, hos_type, pv, city, add, deps, lat, lon
        FROM testhosp
    """
    df = pd.read_sql(query, engine)
    return df

# ─────────────── 유틸 함수 ───────────────
def get_lat_lon(address):
    url = "https://dapi.kakao.com/v2/local/search/address.json"
    params = {"query": address}
    res = requests.get(url, headers=headers, params=params)
    docs = res.json().get("documents")
    if docs:
        return float(docs[0]["y"]), float(docs[0]["x"])
    return None

def vectorized_haversine(lat1, lon1, lat2s, lon2s):
    R = 6371
    dlat = np.radians(lat2s - lat1)
    dlon = np.radians(lon2s - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2s)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def match_exact_departments(treatment, selected_depts):
    if pd.isna(treatment):
        return False
    return any(dept.strip() in treatment.split(",") for dept in selected_depts)

# ─────────────── 지도 및 병원 목록 출력 ───────────────
def show_map_and_list(radius, df_filtered):
    center_lat, center_lon = st.session_state.get("focused_location", (37.5665, 126.9780))
    df2 = df_filtered.dropna(subset=["lat","lon"]).copy()
    df2["distance"] = vectorized_haversine(center_lat, center_lon, df2["lat"].values, df2["lon"].values)
    nearby = df2[df2["distance"] <= radius].sort_values("distance").reset_index(drop=True)

    m = folium.Map(location=(center_lat, center_lon), zoom_start=16)
    cluster = MarkerCluster().add_to(m)
    folium.Marker(
        location=(center_lat, center_lon),
        tooltip="선택 위치",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(cluster)
    for row in nearby.itertuples():
        folium.Marker(
            location=(row.lat, row.lon),
            tooltip=row.hospital_name,
            popup=folium.Popup(
                f"<strong style='color:black'>{row.hospital_name}</strong><br>"
                f"<span style='color:black'>{row.address}<br>{row.treatment}</span>",
                max_width=250
            ),
            icon=folium.Icon(color="blue")
        ).add_to(cluster)

    map_col, list_col = st.columns([3,2])
    with map_col:
        st_folium(m, width=700, height=450)
    with list_col:
        st.header("📋 병원 목록")
        if nearby.empty:
            st.info("❌ 조건에 맞는 병원이 없습니다.")
            return
        visible = st.session_state.get("visible_count", 3)
        for row in nearby.iloc[:visible].itertuples():
            lat, lon = row.lat, row.lon
            st.markdown(f"""
<div style="background:white;padding:12px;border-radius:8px;margin-bottom:8px;">
  <strong style="color:black;font-size:16px;">{row.hos_nm}</strong><br>
  <span style="font-size:13px;color:#333;">
    주소: {row.add}<br>
    진료과: {row.deps}<br>
    거리: {row.distance:.2f} km
  </span>
  <div style="display:flex;gap:6px;margin-top:8px;">
    <a href="https://map.kakao.com/link/map/{lat},{lon}" target="_blank" style="text-decoration:none;">
      <button style="display:flex;align-items:center;background:#FFEB00;color:black;border:none;padding:6px 12px;border-radius:5px;">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/kakaotalk.svg"
             alt="카카오맵" style="width:16px;height:16px;margin-right:4px;"/>카카오맵
      </button>
    </a>
    <a href="https://map.naver.com/v5/search/{lat},{lon}" target="_blank" style="text-decoration:none;">
      <button style="display:flex;align-items:center;background:#03C75A;color:white;border:none;padding:6px 12px;border-radius:5px;">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/naver.svg"
             alt="네이버지도" style="width:16px;height:16px;margin-right:4px;"/>네이버지도
      </button>
    </a>
    <a href="https://www.google.com/maps/search/?api=1&query={lat},{lon}" target="_blank" style="text-decoration:none;">
      <button style="display:flex;align-items:center;background:#4285F4;color:white;border:none;padding:6px 12px;border-radius:5px;">
        <img src="https://cdn.jsdelivr.net/npm/simple-icons@v9/icons/google.svg"
             alt="구글지도" style="width:16px;height:16px;margin-right:4px;"/>구글지도
      </button>
    </a>
  </div>
</div>
""", unsafe_allow_html=True)
        if visible < len(nearby) and st.button("📄 병원 더보기"):
            st.session_state["visible_count"] = visible + 3
            st.rerun()

# ─────────────── 주소 입력 처리 ───────────────
def render_address_input(df_filtered, radius):
    addr = st.text_input("도로명 주소 입력", "서울특별시 광진구 능동로 120")
    if addr:
        loc = get_lat_lon(addr)
        if loc:
            st.session_state["focused_location"] = loc
            show_map_and_list(radius, df_filtered)
        else:
            st.warning("❌ 주소를 찾을 수 없습니다.")

# ─────────────── GPS 처리 ───────────────
def render_gps_location(df_filtered, radius):
    if "gps_location" not in st.session_state:
        with st.spinner("📡 위치 정보를 가져오는 중입니다..."):
            st.session_state["gps_location"] = get_geolocation()
    loc = st.session_state.get("gps_location")
    coords = loc.get("coords") if loc else None

    if coords:
        lat, lon = coords["latitude"], coords["longitude"]
        acc = coords.get("accuracy",9999)
        st.info(f"📍 현재 위치 정확도: ±{int(acc)}m")
        if acc <= 100:
            st.session_state["focused_location"] = (lat, lon)
            show_map_and_list(radius, df_filtered)
        else:
            st.warning(f"⚠️ 위치 정확도가 낮습니다. (±{int(acc)}m)")
            if st.button("📍 주소 입력으로 전환"):
                st.session_state["location_method"] = "주소 입력"
                st.rerun()
    else:
        st.warning("⚠️ 위치 정보를 가져올 수 없습니다.")
    if st.button("🔄 위치 다시 요청"):
        st.session_state["gps_location"] = get_geolocation()
        st.rerun()

# ─────────────── 메인 ───────────────
st.set_page_config(page_title="병원 위치 시각화", layout="wide")
st.title("🏥 병원 위치 시각화 서비스")

if "location_method" not in st.session_state:
    st.session_state["location_method"] = "현재 위치(GPS)"
if "visible_count" not in st.session_state:
    st.session_state["visible_count"] = 3

df = load_hospital_data()

method = st.radio(
    "위치 입력 방식",
    ["현재 위치(GPS)", "주소 입력"],
    index=0 if st.session_state["location_method"]=="현재 위치(GPS)" else 1,
    horizontal=True
)
if method != st.session_state["location_method"]:
    st.session_state["location_method"] = method
    st.rerun()

col1, col2, col3 = st.columns(3)
with col1:
    radius = st.slider("📏 반경 (km)", 0.1, 5.0, 1.0, 0.1)
with col2:
    depts = sorted({d.strip() for t in df["deps"].dropna() for d in t.split(",")})
    selected_depts = st.multiselect("진료과 필터", depts)
with col3:
    search_name = st.text_input("🔍 병원명 필터")

filtered = df.copy()
if selected_depts:
    filtered = filtered[filtered["deps"].apply(lambda t: match_exact_departments(t, selected_depts))]
if search_name:
    filtered = filtered[filtered["hos_nm"].str.contains(search_name, case=False, na=False)]

if st.session_state["location_method"] == "현재 위치(GPS)":
    render_gps_location(filtered, radius)
else:
    render_address_input(filtered, radius)

