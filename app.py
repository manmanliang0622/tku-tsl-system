import streamlit as st
import cv2
import numpy as np
import av
import os
import torch
import json
import queue # 解決 Thread-safe 問題的關鍵
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
from dotenv import load_dotenv

# 1. 環境設定
load_dotenv()
st.set_page_config(
    page_title="淡江大學 TSL 雙向手語轉譯系統",
    page_icon="🤟",
    layout="wide"
)

# 自動偵測設備 (解決雲端無 GPU 會崩潰的問題)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 初始化 Queue 與 Session State
# WebRTC 回呼函數不能直接寫入 st.session_state，必須透過 Queue 傳遞結果
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()
if 'detected_glosses' not in st.session_state:
    st.session_state.detected_glosses = []
if 'translated_chinese' not in st.session_state:
    st.session_state.translated_chinese = ""

# 3. 資源載入 (使用快取)
@st.cache_resource
def load_labels():
    try:
        with open('label_map.json', 'r', encoding='utf-8') as f:
            l_map = json.load(f)
        return {v: k for k, v in l_map.items()}
    except:
        return {0: "測試標籤"}

actions = load_labels()

# 4. MediaPipe 初始化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 用於儲存序列的臨時 buffer (不要放進 session_state 以免回呼函數報錯)
sequence_buffer = []

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = holistic.process(img_rgb)
    
    # 繪圖
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # 模擬預測邏輯 (曼璇，等妳放上核心模型後，將結果 put 進 queue)
    # example: if prediction_confidence > 0.85: st.session_state.result_queue.put(action)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI 介面佈局 ---
st.title("🤟 淡江校園窗口：台灣手語雙向轉譯系統")

tab1, tab2 = st.tabs(["👐 手語轉中文", "👤 中文轉手語"])

with tab1:
    col_cam, col_res = st.columns([3, 2])
    with col_cam:
        webrtc_streamer(
            key="tsl-system",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )
        if st.button("🧹 清除辨識結果"):
            st.session_state.detected_glosses = []
            st.rerun()

    with col_res:
        st.subheader("📝 翻譯結果")
        
        # 從 Queue 中提取背景辨識到的資料並更新 UI
        while not st.session_state.result_queue.empty():
            new_val = st.session_state.result_queue.get()
            if not st.session_state.detected_glosses or new_val != st.session_state.detected_glosses[-1]:
                st.session_state.detected_glosses.append(new_val)

        st.write("**● 偵測到的手語詞彙 (Glosses)**")
        st.info(" ➔ ".join(st.session_state.detected_glosses) if st.session_state.detected_glosses else "等待偵測中...")

with tab2:
    st.write("（未來可對接 SMPL-X 虛擬人生成預覽）")
