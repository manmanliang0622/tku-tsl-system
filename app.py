import streamlit as st
import cv2
import numpy as np
import time
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp

# 1. 頁面基礎設定
st.set_page_config(
    page_title="淡江大學 TSL 雙向手語轉譯系統",
    page_icon="🤟",
    layout="wide"
)

# 2. 現代感 CSS
st.markdown("""
    <style>
    .stApp { background-color: #F8FAFC; }
    
    .main-header {
        background: linear-gradient(135deg, #005696 0%, #00AEEF 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 86, 150, 0.2);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 50px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: transparent;
        padding: 0px 30px;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #005696 !important;
        border-bottom: 3px solid #005696 !important;
    }

    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        border-left: 6px solid #005696;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        min-height: 80px;
        display: flex;
        align-items: center;
    }
    .empty-placeholder {
        color: #94A3B8;
        font-style: italic;
    }
    
    .video-placeholder {
        border: 2px dashed #CBD5E1; 
        border-radius: 15px; 
        height: 320px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        background-color: white;
        box-shadow: inset 0 2px 4px 0 rgba(0,0,0,0.02);
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. 初始化 Session State
if 'detected_gloss' not in st.session_state:
    st.session_state.detected_gloss = ""
if 'translated_chinese' not in st.session_state:
    st.session_state.translated_chinese = ""
if 'video_ready' not in st.session_state:
    st.session_state.video_ready = False

# 4. 初始化 MediaPipe 與 WebRTC 設定
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 設定 Holistic 模型 (提取身體與雙手關鍵點)
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# WebRTC 伺服器設定 (確保雲端連線穩定)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 影像處理回呼函式 (即時畫上骨架)
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # 轉換成 OpenCV 格式並翻轉
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    
    # 轉換色彩空間供 MediaPipe 讀取
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 進行骨架偵測
    results = holistic.process(img_rgb)
    
    # 在影像上畫出骨架連線
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 頂部標題區 ---
st.markdown("""
    <div class="main-header">
        <h1 style='margin:0; font-size: 2.3rem; font-weight: 800;'>淡江校園窗口與行政服務：台灣手語雙譯系統</h1>
        <p style='margin:10px 0 0 0; opacity: 0.9;'>Tamkang University - AI Sign Language Bidirectional Translation</p>
    </div>
    """, unsafe_allow_html=True)

# --- 分頁鍵 ---
tab_slr, tab_slp = st.tabs(["👐 手語 -> 中文", "👤 中文 -> 手語"])

# ==========================================
# --- 模式一：手語 -> 中文 ---
# ==========================================
with tab_slr:
    col_cam, col_res = st.columns([3, 2])
    
    with col_cam:
        st.markdown("### 📹 雲端即時影像串流")
        st.caption("🔍 點擊下方 START 按鈕啟動瀏覽器相機並進行骨架偵測...")
        
        # WebRTC 視訊串流區塊
        webrtc_streamer(
            key="tsl-scanner",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with col_res:
        st.markdown("### 🧠 翻譯")
        
        st.write("**● 偵測到的手語**")
        if st.session_state.detected_gloss:
            st.info(st.session_state.detected_gloss)
        else:
            st.markdown('<p class="empty-placeholder">尚無數據...</p>', unsafe_allow_html=True)
        
        st.write("") 
        
        st.write("**● 中文翻譯**")
        if st.session_state.translated_chinese:
            st.markdown(f"""
                <div class="result-card">
                    <h4 style="margin:0; color:#1E293B;">{st.session_state.translated_chinese}</h4>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-card"><p class="empty-placeholder">等待翻譯...</p></div>', unsafe_allow_html=True)

# ==========================================
# --- 模式二：中文 -> 手語 ---
# ==========================================
with tab_slp:
    col_input, col_avatar = st.columns([2, 3])
    
    with col_input:
        st.markdown("### ⌨️ 中文文字輸入")
        user_input = st.text_area("請輸入中文內容：", height=200, placeholder="例如：請攜帶學生證到商管大樓辦理。")
        
        if st.button("🪄 生成手語動作"):
            if user_input:
                with st.spinner("LLM 正在解析語法與生成骨架動畫..."):
                    time.sleep(1.5) 
                    st.session_state.video_ready = True
                    st.rerun()
            else:
                st.warning("⚠️ 請先輸入中文內容再點擊生成喔！")

        if st.session_state.video_ready:
            st.write("**TSL 動作序列：**")
            st.code("同學 | 攜帶 | 學生證 | 商管大樓 | 辦理", language=None)
            
            if st.button("🔄 清除並重新生成"):
                st.session_state.video_ready = False
                st.rerun()

    with col_avatar:
        if st.session_state.video_ready:
            # 目前以測試動畫代替，未來可換成妳算繪好的虛擬人 MP4
            st.video("https://www.w3schools.com/html/mov_bbb.mp4")
            st.success("✅ 動畫生成完畢！")
            st.caption("💡 提示：此動畫由 TAIDE 解析語意後，透過骨架驅動生成。")
        else:
            st.markdown("""
                <div class="video-placeholder">
                    <div style="text-align: center; color: #94A3B8;">
                        <span style="font-size: 2.5rem;">🎞️</span><br>
                        <span style="font-weight: 500; font-size: 1.1rem;">虛擬人生成的影片將在此框內播放</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            st.write("")
            st.caption("💡 提示：系統會根據翻譯出的 Gloss 序列驅動虛擬人骨架")

# --- 頁尾 ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: #94A3B8;'>淡江大學 資訊管理學系 大專生計畫 - 曼璇 製作</p>", unsafe_allow_html=True)
