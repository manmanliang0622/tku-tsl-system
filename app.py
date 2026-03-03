import streamlit as st
import cv2
import numpy as np
import av
import queue
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp

# 1. 頁面基礎設定
st.set_page_config(
    page_title="淡江大學 TSL 雙向手語轉譯系統",
    page_icon="🤟",
    layout="wide"
)

# 2. 注入現代化 CSS 樣式
st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; }
    .main-header {
        background: linear-gradient(135deg, #005696 0%, #00AEEF 100%);
        padding: 2rem; border-radius: 20px; color: white;
        text-align: center; margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 86, 150, 0.2);
    }
    .result-card {
        background-color: white; padding: 25px; border-radius: 18px;
        border-left: 8px solid #005696; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-top: 1rem;
    }
    .status-tag {
        background-color: #E2E8F0; padding: 5px 12px;
        border-radius: 20px; font-size: 0.85rem; color: #475569;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. 初始化線程安全的資料佇列
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()
if "detected_glosses" not in st.session_state:
    st.session_state.detected_glosses = []

# 4. MediaPipe 初始化 (強制 CPU 模式，解決 0x3008 錯誤)
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 使用快取確保單一實例
@st.cache_resource
def get_holistic():
    return mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,         # 降低複雜度以提升 CPU 效能
        enable_segmentation=False,  # 關閉不必要的計算
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

holistic = get_holistic()

# 5. 影像處理回呼函數
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1) # 鏡像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 執行 MediaPipe 偵測
    results = holistic.process(img_rgb)
    
    # 繪製骨架預覽 (這是讓簡報加分的視覺效果)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    # 模擬辨識邏輯：偵測到手部時，傳入測試詞彙至 Queue
    if results.left_hand_landmarks or results.right_hand_landmarks:
        # 這裡未來換成妳的 TSLModel 推理邏輯
        # st.session_state.result_queue.put("偵測中...") 
        pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI 介面佈置 ---
st.markdown('<div class="main-header"><h1>淡江大學 TSL 雙向手語轉譯系統</h1><p>校園行政窗口服務 · 台灣手語 AI 即時辨識</p></div>', unsafe_allow_html=True)

tab_slr, tab_slp = st.tabs(["👋 手語轉中文 (辨識模式)", "🤖 中文轉手語 (生成模式)"])

with tab_slr:
    col_cam, col_res = st.columns([3, 2])
    
    with col_cam:
        st.markdown('<span class="status-tag">📷 LIVE CAMERA</span>', unsafe_allow_html=True)
        webrtc_streamer(
            key="tsl-main-streamer",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if st.button("🗑️ 清除所有辨識結果"):
            st.session_state.detected_glosses = []
            st.rerun()

    with col_res:
        st.subheader("📝 翻譯與狀態偵測")
        
        # 定期從 Queue 讀取資料並更新 UI
        while not st.session_state.result_queue.empty():
            res = st.session_state.result_queue.get()
            if not st.session_state.detected_glosses or res != st.session_state.detected_glosses[-1]:
                st.session_state.detected_glosses.append(res)

        with st.container():
            st.write("**● 偵測到的手語序列 (Glosses)**")
            if st.session_state.detected_glosses:
                st.info(" ➔ ".join(st.session_state.detected_glosses))
            else:
                st.markdown('<p style="color:#94A3B8; font-style:italic;">請站在鏡頭前比出手語...</p>', unsafe_allow_html=True)
            
            st.write("**● 淡江行政回覆 (自然中文)**")
            st.markdown('<div class="result-card"><h4>等待 AI 組句中...</h4></div>', unsafe_allow_html=True)

with tab_slp:
    st.info("💡 此功能將串接 SMPL-X 虛擬人模型，根據行政文字生成手語動作影片。")
    st.image("https://via.placeholder.com/800x450.png?text=Avatar+Preview+Placeholder")
