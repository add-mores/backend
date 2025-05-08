import streamlit as st

# 페이지 설정
st.set_page_config(page_title="🩺 증상 기반 질병 예측 시스템", layout="wide")

# 카드 스타일 CSS
st.markdown(
    """
    <style>
    .card {
        background-color: #ffffff;
        color: black; 
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
        opacity: 0;  /* 초기에는 숨김 상태 */
        animation: fadein 1.2s ease-in-out forwards;
    }
    .card:hover {
        transform: scale(1.02);
    }
    .center {
        text-align: center;
    }
    .intro-box {
        text-align: center;
        padding: 30px 0 10px 0;
        font-family: 'Segoe UI', sans-serif;
        animation: fadein 1.2s ease-in-out;
    }
    .intro-box p {
        font-size: 22px;
        color: #333333;
        line-height: 1.6;
    }
    .intro-box strong {
        color: #00796B;
        font-weight: 600;
    }
    @keyframes fadein {
        0% { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 제목 
st.markdown("<h1 class='center'>🩺 증상 기반 질병 예측 및 의료 정보 추천 서비스</h1>", unsafe_allow_html=True)

# 소개 문구
st.markdown(
    """
    <div class='intro-box'>
        <p><strong>입력한 증상으로 AI가 유사한 질병을 예측하고,</strong><br>
        <strong>관련 병원과 약품 정보를 통합적으로 추천해드립니다.</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# 기능 안내 카드
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <a href='/disease' target='_self' style='text-decoration: none;'>
            <div class='card'>
                <h3>🔍 질병 추천</h3>
                <p>증상을 기반으로 유사한 질병을 추천합니다.</p>
            </div>
        </a>
        """, unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <a href='/hospital' target='_self' style='text-decoration: none;'>
            <div class='card'>
                <h3>🏥 병원 찾기</h3>
                <p>현 위치 또는 지역 기반 병원을 추천합니다.</p>
            </div>
        </a>
        """, unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <a href='/medicine' target='_self' style='text-decoration: none;'>
            <div class='card'>
                <h3>💊 약 추천</h3>
                <p>질병/증상에 맞는 의약품 정보를 제공합니다.</p>
            </div>
        </a>
        """, unsafe_allow_html=True
    )

st.markdown("---")

st.write(" ")
st.write(" ")

# 경고 문구 (일반 텍스트로 페이지 하단에)
st.write("❗ 이용 시 유의사항")
st.markdown(
    """
    <div style="font-size:13px; line-height:1.6">
    이 웹사이트에서 제공하는 모든 정보는 학습 및 일반적인 정보 제공을 목적으로 하며, 의학적 진단이나 치료를 대체하지 않습니다.<br>
    건강에 관한 의문이 있을 경우, 반드시 의료 전문가인 의사의 진단을 받으시기 바랍니다.<br>
    또한 당사는 의약품 및 건강 관련 정보의 정확성을 보장하지 않으며,<br>
    의약품 사용과 관련된 모든 결정은 의료 전문가의 지도 아래에서 이루어져야 합니다.<br>
    이 웹사이트는 의료 서비스를 제공하지 않으며, 의약품 판매를 목적으로 하지 않습니다.
    </div>
    """,
    unsafe_allow_html=True
)
