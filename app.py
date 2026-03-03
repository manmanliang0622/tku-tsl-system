import streamlit as st
import cv2
import numpy as np
import av
import os
import torch
import json
import queue # 用於解決執行緒安全通訊
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
from dotenv import load_dotenv

# 載入核心組件
from core.data_loader import TSLModel 
from core.tsl_smart_translator import translate_tsl_to_formal_chinese

# 1. 頁面基礎設定
load_dotenv()
st.set_page_config(
    page_title="淡江大學 TSL 雙向手語轉譯系統",
    page_icon="🤟",
    layout="wide"
)

# 自動偵測設備，確保在雲端 CPU 環境也能執行
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. 共享狀態管理：建立一個 Thread-safe 的佇列
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue()
if 'detected_glosses' not in st.session_state:
    st.session_state.detected_glosses = []
if 'translated_chinese' not in st.session_state:
    st.session_state.translated_chinese = ""

# 3. 資源載入 (使用快取)
@st.cache_resource
def load_all_resources():
    with open('label_map.json', 'r', encoding='utf-8') as f:
        l_map = json.load(f)
    actions_map = {v: k for k, v in l_map.items()}
    
    # 載入模型
    model = TSLModel(input_size=126, hidden_size=128, num_layers=2, num_classes=len(actions_map))
    if os.path.exists('models/tsl_model.pth'):
        model.load_state_dict(torch.load('models/tsl_model.pth', map_location=device))
    model.to(device).eval()
    return model, actions_map

tsl_ai_model, actions = load_all_resources()

# 4. MediaPipe 初始化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 用於儲存特徵序列的臨時 buffer
sequence_buffer = []

# WebRTC 影像處理回呼 (注意：此函數在獨立執行緒運行，不可直接存取 session_state)
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

    # 提取特徵
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    keypoints = np.concatenate([lh, rh])

    sequence_buffer.append(keypoints)
    if len(sequence_buffer) > 30:
        sequence_buffer.pop(0)

    if len(sequence_buffer) == 30:
        input_tensor = torch.FloatTensor(np.expand_dims(sequence_buffer, axis=0)).to(device)
        with torch.no_grad():
            res = tsl_ai_model(input_tensor)
            prob = torch.softmax(res, dim=1)
            max_prob, idx = torch.max(prob, dim=1)
            
            if max_prob.item() > 0.85:
                # 關鍵：將結果放入 Queue，由主執行緒讀取更新 UI
                st.session_state.result_queue.put(actions[idx.item()])

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI 介面 ---
st.title("🤟 淡江校園窗口：台灣手語雙譯系統")

tab_slr, tab_slp = st.tabs(["👋 手語轉中文", "🤖 中文轉手語"])

with tab_slr:
    col_cam, col_res = st.columns([3, 2])
    with col_cam:
        webrtc_streamer(
            key="tsl-scanner",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_frame_callback=video_frame_callback,
            async_processing=True,
        )
        if st.button("🧹 清除辨識結果"):
            st.session_state.detected_glosses = []
            st.session_state.translated_chinese = ""
            st.rerun()

    with col_res:
        st.subheader("📝 翻譯與轉譯結果")
        
        # 從 Queue 中提取背景偵測到的詞彙並更新 UI
        while not st.session_state.result_queue.empty():
            new_gloss = st.session_state.result_queue.get()
            if not st.session_state.detected_glosses or new_gloss != st.session_state.detected_glosses[-1]:
                st.session_state.detected_glosses.append(new_gloss)
                # 自動觸發翻譯邏輯
                st.session_state.translated_chinese = translate_tsl_to_formal_chinese(st.session_state.detected_glosses)

        st.write("**● 偵測到的手語序列**")
        st.info(" ➔ ".join(st.session_state.detected_glosses) if st.session_state.detected_glosses else "等待偵測...")
        
        st.write("**● 淡江行政回覆 (Gemini 轉譯)**")
        if st.session_state.translated_chinese:
            st.success(st.session_state.translated_chinese)
        else:
            st.markdown('<p style="color:gray;">等待 AI 組句...</p>', unsafe_allow_html=True)
